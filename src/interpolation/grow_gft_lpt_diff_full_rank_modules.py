from transformers import AutoTokenizer, pipeline
import eval_utils
from grow import grow_depth, grow_width
import json
import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator
from accelerate import Accelerator
import gc
import os
from peft import LoraConfig, get_peft_model

import operator
from grow_utils import *


class args:
    small_pretrained_model_cpt = 'EleutherAI/pythia-410m'
    small_finetuned_model_cpt = 'models/pythia_410m_r=8_0.0001_gsm8k'
    large_pretrained_model_cpt = 'EleutherAI/pythia-1.4b'
    large_finetuned_model_cpt = 'models/pythia_1.4b_r=8_0.0001_gsm8k'

    batch_size = 1
    rank = 8
    lora_alpha = 32
    target_modules = ["query_key_value"]

    eval_file = "eval/eval_lora_full_rank_modules.json"


def add_diff_as_full_rank(base_model, diff_state_dict):
    """
    Add the difference between the base model and the grown model as full rank to the base model
    """

    # Add the difference to the base model
    for key, value in base_model.state_dict().items():
        base_model.state_dict()[key] = value + diff_state_dict[key]


def add_diff_as_full_rank_modules(base_model, diff_state_dict, target_modules):
    """
    Add the difference between the base model and the grown model as full rank to the base model
    """

    # Add the difference to the base model for the target modules
    for key, value in diff_state_dict.items():
        if any([target_module in key for target_module in target_modules]):
            print(f"Adding difference to module {key}")
            base_model.state_dict()[key] = base_model.state_dict()[key] + value

    base_model.load_state_dict(base_model.state_dict())




def add_diff_as_lora(base_model, diff_state_dict, target_modules=["query_key_value"]):
    """
    Add the difference between the base model and the grown model as a lora adapter to the base model
    """

    # For all the target modules in the diff_state_dict, update the lora adapter of the base model

    for key, value in diff_state_dict.items():
        base_model_key = '.'.join(key.split(".")[:-1])
        weight_name = key.split(".")[-1]

        if weight_name == 'weight' and any([target_module in key for target_module in target_modules]):

            # print('-' * 50)
            # print(f"Module {key} has a corresponding lora adapter")

            # Convert the value to approximate product of low rank matrices
            U, S, Vh = torch.linalg.svd(value.T, full_matrices=False)

            # Reduce to rank args.rank
            lora_A = (U[:, :args.rank] @ torch.diag(S[:args.rank])).T.contiguous()
            lora_B = (Vh[:args.rank, :]).T.contiguous()

            # Update value module base_model_key.lora_A and base_model_key.lora_B
            parent_module = base_model
            for attr in base_model_key.split("."):
                parent_module = getattr(parent_module, attr)

            # print('Old values of lora_A and lora_B')
            # print(f"lora_A: {parent_module.lora_A['default'].weight}")
            # print(f"lora_B: {parent_module.lora_B['default'].weight}")

            # Copy the values from lora_A and lora_B to the parent 
            # module's lora_A and lora_B modules
            # print(f"lora_A old shape: {parent_module.lora_A['default'].weight.shape}")
            # print(f"lora_B old shape: {parent_module.lora_B['default'].weight.shape}")

            parent_module.lora_A['default'].weight = torch.nn.Parameter(lora_A)
            parent_module.lora_B['default'].weight = torch.nn.Parameter(lora_B)

            # print(f"lora_A new shape: {parent_module.lora_A['default'].weight.shape}")
            # print(f"lora_B new shape: {parent_module.lora_B['default'].weight.shape}")
            # print('-' * 50)



            # print('New values of lora_A and lora_B set')
            # print(f"lora_A: {parent_module.lora_A['default'].weight}")
            # print(f"lora_B: {parent_module.lora_B['default'].weight}")
            # print('-' * 50)



def playground():

    small_finetuned_model = eval_utils.load_model(args.small_pretrained_model_cpt, args.small_finetuned_model_cpt)


    # Grow the small finetuned model to 1.4b
    grown_finetuned_model = grow_model_410m_to_1_4b(small_finetuned_model, random_noise=False)


    # Delete small_finetuned_model
    del small_finetuned_model
    clean_memory()

    large_pretrained_model = eval_utils.load_model(args.large_pretrained_model_cpt)

    # Move large_pretrained_model to CPU
    large_pretrained_model.cpu()

    # Find difference between model parameters of grown_finetuned_model and large_pretrained_model
    large_small_diff_state_dict = get_model_diff(large_pretrained_model, grown_finetuned_model)

    # Delete grown_finetuned_model
    del grown_finetuned_model
    clean_memory()

    
    # Load the empty large fine-tuned model
    large_pretrained_model = eval_utils.load_model(args.large_pretrained_model_cpt)

    # Add the difference (in full rank) to the large pretrained model
    add_diff_as_full_rank_modules(
        base_model=large_pretrained_model,
        diff_state_dict=large_small_diff_state_dict,
        target_modules=args.target_modules
    )

    

            


def main():
    eval_results = {}

    # Define accelerator
    accelerator = Accelerator()
    device = accelerator.device
    print(f"device: {device}")


    with accelerator.main_process_first():

        # Get the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.small_pretrained_model_cpt)

        # Test dataloader
        _, eval_dataloader = get_dataloader("gsm8k", batch_size=args.batch_size)

        # Clean up memory
        del _
        clean_memory()


        print("Loading models...")

        # 1. Load and evaluate the large pretrained model

        # Load the large pretrained model
        large_pretrained_model = eval_utils.load_model(args.large_pretrained_model_cpt)


    # Prepare the models for evaluation
    large_pretrained_model, tokenizer, eval_dataloader = accelerator.prepare(
        large_pretrained_model,
        tokenizer,
        eval_dataloader
    )

    # Print memory usage
    print("Memory usage after loading large pretrained model")
    eval_utils.print_memory_usage()

    # Evaluate the large pretrained model
    avg_eval_loss, metrics = eval_utils.evaluate_model(
        large_pretrained_model, eval_dataloader, accelerator, tokenizer
    )

    eval_results["large_pretrained_model"] = {
        "model_name": args.large_pretrained_model_cpt,
        "avg_eval_loss": avg_eval_loss,
        **metrics
    }

    print("Evaluated large pretrained model")
    print(eval_results["large_pretrained_model"])

    # Move large_pretrained_model to CPU
    large_pretrained_model.cpu()
    clean_memory()

    # Print memory usage after moving large_pretrained_model to CPU
    print("Memory usage after moving large pretrained model to CPU")
    eval_utils.print_memory_usage()

    # 2. Grow small model and find difference between large_pretrained_model and grown small model

    # Load small fine-tuned model
    with accelerator.main_process_first():
        small_finetuned_model = eval_utils.load_model(args.small_pretrained_model_cpt, args.small_finetuned_model_cpt)

    # Grow the small finetuned model to 1.4b
    grown_finetuned_model = grow_model_410m_to_1_4b(small_finetuned_model, random_noise=False)


    # Delete small_finetuned_model
    del small_finetuned_model
    clean_memory()

    # Find difference between model parameters of grown_finetuned_model and large_pretrained_model
    large_small_diff_state_dict = get_model_diff(large_pretrained_model, grown_finetuned_model)

    # Delete grown_finetuned_model
    del grown_finetuned_model
    clean_memory()


    # Add the difference (in full rank) to the large pretrained model
    add_diff_as_full_rank_modules(
        base_model=large_pretrained_model,
        diff_state_dict=large_small_diff_state_dict,
        target_modules=args.target_modules
    )

    # Save large_pretrained_model to disk
    large_pretrained_model.save_pretrained("tmp_large_pretrained_model")

    # Clean up memory
    del large_pretrained_model
    clean_memory()

    # Load the large_full_rank_model from disk
    large_full_rank_model = eval_utils.load_model("tmp_large_pretrained_model")


    # Delete the temporary directory
    for file in os.listdir("tmp_large_pretrained_model"):
        os.remove(os.path.join("tmp_large_pretrained_model", file))
    os.rmdir("tmp_large_pretrained_model")

    # Prepare the models for evaluation
    large_full_rank_model = accelerator.prepare(
        large_full_rank_model,
    )

    # Print memory usage
    print("Memory usage after growing difference")
    eval_utils.print_memory_usage()

    # Evaluate the large_full_rank_model with the grown difference
    avg_eval_loss, metrics = eval_utils.evaluate_model(
        large_full_rank_model, eval_dataloader, accelerator, tokenizer
    )

    eval_results["large_full_rank_model_diff"] = {
        "model_name": args.large_pretrained_model_cpt,
        "avg_eval_loss": avg_eval_loss,
        **metrics
    }

    print("Evaluated grown small finetuned model with grown full-rank difference")
    print(eval_results["large_full_rank_model_diff"])

    del large_full_rank_model
    clean_memory()

    # 3. Load and evaluate the large finetuned model

    with accelerator.main_process_first():

        # Load the large finetuned model
        large_finetuned_model = eval_utils.load_model(args.large_pretrained_model_cpt, args.large_finetuned_model_cpt)
        large_finetuned_model.gradient_checkpointing_enable()

    # Prepare the models for evaluation
    large_finetuned_model = accelerator.prepare(
        large_finetuned_model
    )
    
    # Evaluate the large finetuned model
    avg_eval_loss, metrics = eval_utils.evaluate_model(
        large_finetuned_model, eval_dataloader, accelerator, tokenizer
    )

    eval_results["large_finetuned_model"] = {
        "model_name": args.large_finetuned_model_cpt,
        "avg_eval_loss": avg_eval_loss,
        **metrics
    }

    print("Evaluated large finetuned model")
    print(eval_results["large_finetuned_model"])

    # Save the evaluation results to eval_results.json

    with open(args.eval_file, "w") as f:
        json.dump(eval_results, f, indent=4)
    



if __name__ == "__main__":


    main()
    # playground()