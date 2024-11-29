from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from grow import grow_depth, grow_width
from typing import Union, List
import evaluate
from accelerate import Accelerator
import psutil
import subprocess
from lm_eval import evaluator

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GPTNeoXForCausalLM
from peft import PeftModel, get_peft_model
from datasets import load_metric


def print_memory_usage():
    print()
    # GPU memory usage (if CUDA is available)
    if torch.cuda.is_available():

        # print(torch.cuda.memory_summary())

        # Run nvidia-smi on the command line to get the GPU memory usage
        
        # Call nvidia-smi and capture the output
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Print the output (stdout contains the nvidia-smi output)
        print(result.stdout)


        # gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**2)  # Convert to MB
        # gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**2)  # Convert to MB
        # print("CUDA device : %s" % torch.cuda.get_device_name(0))
        # print(f"GPU Memory Allocated: {gpu_memory_allocated:.2f} MB")
        # print(f"GPU Memory Reserved: {gpu_memory_reserved:.2f} MB")
    else:
        print("CUDA is not available. GPU memory not in use.")

    # CPU memory usage
    cpu_memory = psutil.virtual_memory()
    print(f"CPU Memory Used: {cpu_memory.used / (1024**2):.2f} MB")
    print(f"Total CPU Memory: {cpu_memory.total / (1024**2):.2f} MB")
    print(f"Available CPU Memory: {cpu_memory.available / (1024**2):.2f} MB")


def generate_instruction_answer(
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        prompt: Union[str, List[str]],
        max_length: int = 128
    ) -> List[str]:
    """
    Run a generation loop till max length and do a weighted sum of the logits of each token
    before doing an argmax to get the token

    Args:
        model_1: First model
        model_2: Second model
        alpha: Interpolation ratio
    """
    # Tokenize the list of prompts
    tokenizer.pad_token = tokenizer.eos_token
    tokenized = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    input_ids = tokenized['input_ids']
    attn_mask = tokenized['attention_mask']

    gen_tokens_batch = []


    # Distribute inference to multiple GPUs if available
    model = torch.nn.DataParallel(model)


    for _ in range(max_length):
        model_pred = model(input_ids=input_ids, attention_mask=attn_mask)

        next_token_logits = model_pred[0][:, -1, :]

        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Append the generated tokens for all batches (batch_size, max_seq_len)
        gen_tokens_batch.append(next_tokens)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        attn_mask = torch.cat([attn_mask, torch.ones_like(attn_mask[:, :1])], dim=-1)

    # Decode the generated tokens
    gen_tokens_batch = torch.stack(gen_tokens_batch, dim=1)[:, :, 0]

    # Pad all tokens after eos with eos
    eos_mask = gen_tokens_batch == tokenizer.eos_token_id
    eos_mask = eos_mask.cumsum(dim=1) > 0

    gen_tokens_batch[eos_mask] = tokenizer.eos_token_id

    generated_text = tokenizer.batch_decode(gen_tokens_batch, skip_special_tokens=True)

    return generated_text


def generate_example_callback(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str):
        
        if not torch.cuda.is_available():
            model = model.to(torch.float32)

        # Output of model
        print('-'*50)
        print('Model output:')
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        output = generator(prompt, max_length=128, num_return_sequences=1, truncation=True, pad_token_id=tokenizer.eos_token_id)[0]['generated_text']
        print(output)



def load_model(model_name: str, path: str = None, retain_peft: bool = False):
    """
    Load a model from a path and model name

    Args:
        model_name: Name of the model
        path: Path to the peft model
    """

    print(f"Loading model...")

    base_model = GPTNeoXForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        # return_dict=True,
        # torch_dtype=torch.float16,
        device_map="auto",
    )
    if path is not None:
        # model = PeftModel.from_pretrained(base_model, path)
        base_model = PeftModel.from_pretrained(base_model, path)

        if not retain_peft:
            base_model = base_model.merge_and_unload()
    
    return base_model
    

def plot_losses(alphas, losses, save_path="interpolation_loss.png", **kwargs):

    title = kwargs.get('title', 'Model Evaluation Loss During Interpolation')
    xlabel = kwargs.get('xlabel', 'Interpolation Alpha (fraction of grown model)')
    ylabel = kwargs.get('ylabel', 'Loss')

    plt.figure(figsize=(10, 5))
    plt.plot(alphas, losses, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(save_path)


def flatten_rouge_scores(rouge_scores):
    flattened_scores = {}
    for score_type, aggregate_score in rouge_scores.items():
        for stat in ['precision', 'recall', 'fmeasure']:
            flattened_scores[f'{score_type}_low_{stat}'] = getattr(aggregate_score.low, stat)
            flattened_scores[f'{score_type}_mid_{stat}'] = getattr(aggregate_score.mid, stat)
            flattened_scores[f'{score_type}_high_{stat}'] = getattr(aggregate_score.high, stat)
    return flattened_scores




def evaluate_model(model, test_dl, accelerator, tokenizer):
    model.eval()
    total_loss = 0
    rouge = evaluate.load('rouge')
    exact_match = evaluate.load('exact_match')

    if accelerator.is_main_process:
        test_iter = tqdm(test_dl, desc="Evaluating")
    else:
        test_iter = test_dl

    for batch in test_iter:
        labels = batch["labels"].detach().cpu().numpy()

        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            generated_ids = torch.argmax(outputs.logits, dim=-1)
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            reference_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            rouge.add_batch(predictions=generated_texts, references=reference_texts)
            exact_match.add_batch(predictions=generated_texts, references=reference_texts)
    
    avg_eval_loss = total_loss / len(test_dl)
    final_rouge_scores = rouge.compute()
    final_exact_match_scores = exact_match.compute()

    if accelerator.is_main_process:
        print(f"Average Evaluation Loss: {avg_eval_loss}")
        print("ROUGE Scores:", final_rouge_scores)
        print("Exact Match Scores:", final_exact_match_scores)

    metrics = {
        **final_rouge_scores,
        **final_exact_match_scores
    }

    return avg_eval_loss, metrics


def evaluate_model_llm_eval(
    model_path: str,
    checkpoint: str,
    tasks: list,
    tokenizer_path: str = None,  # If different from base_model_path
    num_fewshot: int = 0,
    batch_size: int = 4,
    parallelize: bool = False,
):
    """
    Simple evaluation of a model.
    """

    accelerator = Accelerator()
    
    # Load tokenizer
    tokenizer_path = tokenizer_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading model...")
    if checkpoint is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            revision=checkpoint,
            # torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    # Prepare model for evaluation
    model = accelerator.prepare(model)
    
    # Save model to temporary directory
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Saving model to {tmp_dir}...")
        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)
        
        # Run evaluation
        print("Starting evaluation...")
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={tmp_dir},parallelize={parallelize},logits_cache=False",
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            # no_cache=True
        )
    
    return results['results']


# def evaluate_model_loss(model, test_dl, device, tokenizer):
#     model.eval()
#     total_loss = 0

#     for batch in tqdm(test_dl):
#         batch = {k: v.to(device) for k, v in batch.items()}

#         with torch.no_grad():
#             outputs = model(**batch)
#             loss = outputs.loss
#             total_loss += loss.item()

#             # generated_ids = torch.argmax(outputs.logits, dim=-1)
#             # generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#             # reference_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
#             # rouge.add_batch(predictions=generated_texts, references=reference_texts)
    
#     avg_eval_loss = total_loss / len(test_dl)
#     # final_rouge_scores = rouge.compute()
#     # flattened_rouge_scores = flatten_rouge_scores(final_rouge_scores)
#     print(f"Average Evaluation Loss: {avg_eval_loss}")
#     # print("ROUGE Scores:", flattened_rouge_scores)

#     # return avg_eval_loss, flattened_rouge_scores
#     return avg_eval_loss


# def evaluate_model_loss_rouge(model, test_dl, device, tokenizer):
#     model.eval()
#     total_loss = 0
#     rouge = load_metric('rouge')

#     for batch in tqdm(test_dl):
#         batch = {k: v.to(device) for k, v in batch.items()}
#         labels = batch["labels"].detach().cpu().numpy()

#         with torch.no_grad():
#             outputs = model(**batch)
#             loss = outputs.loss
#             total_loss += loss.item()

#             generated_ids = torch.argmax(outputs.logits, dim=-1)
#             generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#             reference_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
#             rouge.add_batch(predictions=generated_texts, references=reference_texts)
    
#     avg_eval_loss = total_loss / len(test_dl)
#     final_rouge_scores = rouge.compute()
#     flattened_rouge_scores = flatten_rouge_scores(final_rouge_scores)
#     print(f"Average Evaluation Loss: {avg_eval_loss}")
#     print("ROUGE Scores:", flattened_rouge_scores)

#     return avg_eval_loss, flattened_rouge_scores
#     # return flattened_rouge_scores
