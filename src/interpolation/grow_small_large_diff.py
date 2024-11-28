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


class args:
    small_pretrained_model_cpt = 'EleutherAI/pythia-410m'
    small_finetuned_model_cpt = 'models/pythia_410m_r=8_0.0001_gsm8k'
    large_pretrained_model_cpt = 'EleutherAI/pythia-1.4b'
    large_finetuned_model_cpt = 'models/pythia_1.4b_r=8_0.0001_gsm8k'

    batch_size = 1
    lora_alpha = 32


if __name__ == "__main__":
    


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

    with accelerator.main_process_first():

        # Load the model
        small_pretrained_model = eval_utils.load_model(args.small_pretrained_model_cpt)


    # Grow the small pretrained model to 1.4b
    grown_small_model = grow_model_410m_to_1_4b(small_pretrained_model, random_noise=True)

    # Find the difference between the large_pretrained_model and the grown_small_model
    large_small_diff_state_dict = get_model_diff(large_pretrained_model, grown_small_model)


    # Clean up memory
    del large_pretrained_model, grown_small_model, small_pretrained_model
    clean_memory()

    # Load small fine-tuned model
    with accelerator.main_process_first():
        small_finetuned_model = eval_utils.load_model(args.small_pretrained_model_cpt, args.small_finetuned_model_cpt)

    # Grow the small finetuned model to 1.4b
    grown_finetuned_model = grow_model_410m_to_1_4b(small_finetuned_model, random_noise=True)

    # Add the difference to the grown_finetuned_model
    new_state_dict = {}

    for key, value in grown_finetuned_model.state_dict().items():
        new_state_dict[key] = value + large_small_diff_state_dict[key]

    # Update the state dict of large_pretrained_model
    grown_finetuned_model.load_state_dict(new_state_dict)

    # Save large_pretrained_model to disk
    grown_finetuned_model.save_pretrained("tmp_large_pretrained_model")

    # Clean up memory
    del large_small_diff_state_dict, new_state_dict, grown_finetuned_model
    clean_memory()

    # Load the grown_finetuned_model from disk
    grown_finetuned_model = eval_utils.load_model("tmp_large_pretrained_model")


    # Delete the temporary directory
    for file in os.listdir("tmp_large_pretrained_model"):
        os.remove(os.path.join("tmp_large_pretrained_model", file))
    os.rmdir("tmp_large_pretrained_model")

    # Prepare the models for evaluation
    grown_finetuned_model = accelerator.prepare(
        grown_finetuned_model,
    )

    # Print memory usage
    print("Memory usage after growing difference")
    eval_utils.print_memory_usage()

    # Evaluate the large_pretrained_model with the grown difference
    avg_eval_loss, metrics = eval_utils.evaluate_model(
        grown_finetuned_model, eval_dataloader, accelerator, tokenizer
    )

    eval_results["grown_finetuned_model_small_large_diff"] = {
        "model_name": args.small_finetuned_model_cpt,
        "avg_eval_loss": avg_eval_loss,
        **metrics
    }

    print("Evaluated grown small finetuned model with grown difference")
    print(eval_results["grown_finetuned_model_small_large_diff"])

    del grown_finetuned_model
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

    with open("eval_results_small_large.json", "w") as f:
        json.dump(eval_results, f, indent=4)
    
    # # Create a shifted_large_pretrained_model by adding the model_diff_state_dict to large_pretrained_model
    # shifted_large_pretrained_model = copy.deepcopy(large_pretrained_model)
    
    # # Add values of model_diff_state_dict to a values of shifted_large_pretrained_model
    # new_state_dict = {}
    # for key, value in shifted_large_pretrained_model.state_dict().items():
    #     new_state_dict[key] = value + finetune_diff_state_dict[key]

    # # Update the state dict of shifted_large_pretrained_model
    # shifted_large_pretrained_model.load_state_dict(new_state_dict)

    # # Test dataloader
    # # eval_dataloader = torch.load("./data/eval_dataloader.pt")
    # _, eval_dataloader = get_dataloader("gsm8k", batch_size=4)

    # # Define accelerator
    # accelerator = Accelerator()


    # # Prepare the models for evaluation
    # large_pretrained_model, shifted_large_pretrained_model, tokenizer = accelerator.prepare(
    #     large_pretrained_model, 
    #     shifted_large_pretrained_model, 
    #     tokenizer
    # )

    # # 1. Interpolate large_pretrained_model and shifted_large_pretrained_model
    # losses, alphas, rouges = Interpolate.interpolate_models_by_weights_loss_rouge(
    #     large_pretrained_model,
    #     shifted_large_pretrained_model,
    #     3,
    #     eval_dataloader,
    #     tokenizer,
    #     accelerator,
    # )
    
    # rouges = [r['rougeL'] for r in rouges]

    # # Plot the losses
    # eval_utils.plot_losses(
    #     alphas, 
    #     losses, 
    #     save_path="interpolation_loss_lpt_slpt.png",
    #     title="Model Evaluation Loss During Interpolation (Large Pretrained Model to Shifted Large Pretrained Model)",
    #     xlabel="Interpolation Alpha (fraction of large pretrained model)",
    #     ylabel="Loss"
    # )

    # # Plot the rouge scores
    # eval_utils.plot_losses(
    #     alphas, 
    #     rouges, 
    #     save_path="interpolation_rouge_lpt_slpt.png",
    #     title="Model Evaluation Rouge Scores During Interpolation (Large Pretrained Model to Shifted Large Pretrained Model)",
    #     xlabel="Interpolation Alpha (fraction of large pretrained model)",
    #     ylabel="Rouge Score"
    # )


    # # Delete unwanted models
    # del large_pretrained_model
    # gc.collect()


    # large_finetuned_model = eval_utils.load_model(args.large_pretrained_model_cpt, args.large_finetuned_model_cpt)


    # # If using CUDA, clear GPU memory cache
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()

    # # Prepare large_finetuned_model for evaluation
    # large_finetuned_model = accelerator.prepare(large_finetuned_model)


    # # 2. Interpolate shifted_large_pretrained_model and large_finetuned_model
    # losses, alphas, rouges = Interpolate.interpolate_models_by_weights_loss_rouge(
    #     shifted_large_pretrained_model,
    #     large_finetuned_model,
    #     3,
    #     eval_dataloader,
    #     tokenizer,
    #     accelerator,
    # )

    # rouges = [r['rougeL'] for r in rouges]

    # # Plot the losses
    # eval_utils.plot_losses(
    #     alphas, 
    #     losses, 
    #     save_path="interpolation_loss_slpt_lft.png", 
    #     title="Model Evaluation Loss During Interpolation (Shifted Large Pretrained Model to Large Finetuned Model)",
    #     xlabel="Interpolation Alpha (fraction of shifted large pretrained model)",
    #     ylabel="Loss"
    # )

    # # Plot the rouge scores
    # eval_utils.plot_losses(
    #     alphas, 
    #     rouges, 
    #     save_path="interpolation_rouge_slpt_lft.png",
    #     title="Model Evaluation Rouge Scores During Interpolation (Shifted Large Pretrained Model to Large Finetuned Model)",
    #     xlabel="Interpolation Alpha (fraction of shifted large pretrained model)",
    #     ylabel="RougeL high Recall Score"
    # )