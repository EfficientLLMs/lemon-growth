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

from interpolate import Interpolate
from grow_utils import *


if __name__ == "__main__":
    
    small_pretrained_model_cpt = 'EleutherAI/pythia-410m'
    small_finetuned_model_cpt = 'models/pythia_410m_r=8_0.0001_gsm8k'
    large_pretrained_model_cpt = 'EleutherAI/pythia-1.4b'
    large_finetuned_model_cpt = 'models/pythia_1.4b_r=8_0.0001_gsm8k'

    batch_size = 1
    eval_file = "eval/eval_grow_diff_raw_also.json"


    eval_results = {}

    # Define accelerator
    accelerator = Accelerator()
    device = accelerator.device
    print(f"device: {device}")


    with accelerator.main_process_first():

        # Get the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(small_pretrained_model_cpt)

        # Test dataloader
        _, eval_dataloader = get_dataloader("gsm8k", batch_size=batch_size)

        # Clean up memory
        del _
        clean_memory()


        print("Loading models...")

        # 1. Load and evaluate the large pretrained model

        # Load the large pretrained model
        large_pretrained_model = eval_utils.load_model(large_pretrained_model_cpt)
        large_pretrained_model.gradient_checkpointing_enable()


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
        "model_name": large_pretrained_model_cpt,
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


    # Evaluate grown small fine-tuned model
    with accelerator.main_process_first():
        small_finetuned_model = eval_utils.load_model(small_pretrained_model_cpt, small_finetuned_model_cpt)


    grown_small_finetuned_model = grow_model_410m_to_1_4b(small_finetuned_model, random_noise=False)

    # Prepare the models for evaluation
    grown_small_finetuned_model = accelerator.prepare(
        grown_small_finetuned_model
    )


    # Evaluate the grown small finetuned model
    avg_eval_loss, metrics = eval_utils.evaluate_model(
        grown_small_finetuned_model, eval_dataloader, accelerator, tokenizer
    )

    eval_results["grown_small_finetuned_model"] = {
        "model_name": small_finetuned_model_cpt,
        "avg_eval_loss": avg_eval_loss,
        **metrics
    }

    # Clean up memory
    grown_small_finetuned_model.cpu()
    del grown_small_finetuned_model
    clean_memory()


    # 2. Grow difference between small pretrained model and small finetuned model to 1.4b
    # and add the difference to the large_pretrained_model

    with accelerator.main_process_first():

        # Load the models
        small_pretrained_model = eval_utils.load_model(small_pretrained_model_cpt)
        # small_finetuned_model = eval_utils.load_model(small_pretrained_model_cpt, small_finetuned_model_cpt)


    # Find the difference between the small_fineturned_model and the small_pretrained_model
    finetune_small_diff_state_dict = get_model_diff(small_finetuned_model, small_pretrained_model)

    # Replace the small_finetuned_model with the difference state dict
    small_finetuned_model.load_state_dict(finetune_small_diff_state_dict)

    # Clean up memory
    small_pretrained_model.cpu()
    del small_pretrained_model
    clean_memory()

    # Print memory usage after loading small pretrained model
    print("Memory usage after loading small pretrained model")
    eval_utils.print_memory_usage()

    # Grow small_finetuned_model to 1.4b (the difference between the two models)
    grown_difference = grow_model_410m_to_1_4b(small_finetuned_model, random_noise=False)

    # Move small_finetuned_model to CPU and free memory
    small_finetuned_model.cpu()
    del small_finetuned_model
    clean_memory()

    # Add the grown difference to the large_pretrained_model without creating a copy
    new_state_dict = {}

    for key, value in large_pretrained_model.state_dict().items():
        new_state_dict[key] = value + grown_difference.state_dict()[key]
    
    # Update the state dict of large_pretrained_model
    large_pretrained_model.load_state_dict(new_state_dict)
    large_pretrained_model.gradient_checkpointing_enable()

    # Save large_pretrained_model to disk
    large_pretrained_model.save_pretrained("tmp_large_pretrained_model")

    # Clean up memory
    del grown_difference, new_state_dict, large_pretrained_model
    clean_memory()

    # Move large_pretrained_model to GPU
    # large_pretrained_model.to(device)

    # Load the large_pretrained_model from disk
    large_pretrained_model = eval_utils.load_model("tmp_large_pretrained_model")

    # Delete the temporary directory
    for file in os.listdir("tmp_large_pretrained_model"):
        os.remove(os.path.join("tmp_large_pretrained_model", file))
    os.rmdir("tmp_large_pretrained_model")

    # Prepare the models for evaluation
    large_pretrained_model = accelerator.prepare(
        large_pretrained_model,
    )

    # Print memory usage
    print("Memory usage after growing difference")
    eval_utils.print_memory_usage()

    # Evaluate the large_pretrained_model with the grown difference
    avg_eval_loss, metrics = eval_utils.evaluate_model(
        large_pretrained_model, eval_dataloader, accelerator, tokenizer
    )

    eval_results["large_pretrained_model_with_grown_diff"] = {
        "model_name": large_pretrained_model_cpt,
        "avg_eval_loss": avg_eval_loss,
        **metrics
    }

    print("Evaluated large pretrained model with grown difference")
    print(eval_results["large_pretrained_model_with_grown_diff"])

    del large_pretrained_model
    clean_memory()

    # 3. Load and evaluate the large finetuned model

    with accelerator.main_process_first():

        # Load the large finetuned model
        large_finetuned_model = eval_utils.load_model(large_pretrained_model_cpt, large_finetuned_model_cpt)
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
        "model_name": large_finetuned_model_cpt,
        "avg_eval_loss": avg_eval_loss,
        **metrics
    }

    print("Evaluated large finetuned model")
    print(eval_results["large_finetuned_model"])

    # Save the evaluation results to eval_results.json

    with open(eval_file, "w") as f:
        json.dump(eval_results, f, indent=4)
    
    