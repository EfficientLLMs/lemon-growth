from transformers import AutoTokenizer, pipeline
import eval_utils
from grow import grow_depth, grow_width
import copy
import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator
from accelerate import Accelerator
import gc

from interpolate import Interpolate

# def grow_model_70_to_410(model_70m):

#     # Depth from 6 -> 24
#     intermediate_grown = grow_depth.expand_layers(model_70m, 6, 12, expand_type='alternate')
#     intermediate_grown = grow_depth.expand_layers(intermediate_grown, 12, 24, expand_type='alternate')

#     # Width from 512 -> 1024
#     model_70m_ft_grown_410m = grow_width.expand_width(intermediate_grown, 512, 1024)

#     return model_70m_ft_grown_410m

def grow_model_410m_to_1_4b(model_410m):

    # # Depth from 6 -> 24
    # intermediate_grown = grow_depth.expand_layers(model_70m, 6, 12, expand_type='alternate')
    # intermediate_grown = grow_depth.expand_layers(intermediate_grown, 12, 24, expand_type='alternate')

    # Width from 512 -> 1024
    model_410m_to_1_4b = grow_width.expand_width(model_410m, 1024, 2048)

    return model_410m_to_1_4b

def get_model_diff(large_model, grown_small_model):
    
    # Find difference between model parameters of large_pretrained_model and small_pretrained_model, 
    # store the difference in a state dict
    model_diff = {}
    for key, value in large_model.state_dict().items():
        model_diff[key] = value - grown_small_model.state_dict()[key]

    return model_diff



def get_dataloader(dataset_name, batch_size=4):

    if dataset_name in ("alpaca_gpt4", "gsm8k"):
        
        train_ds = torch.load(f"./data/{dataset_name}/train.pt")
        eval_ds = torch.load(f"./data/{dataset_name}/eval.pt")

        train_dataloader = DataLoader(
            train_ds,
            batch_size=batch_size,
            collate_fn=default_data_collator,
        )
        eval_dataloader = DataLoader(
            eval_ds,
            batch_size=batch_size,
            collate_fn=default_data_collator,
            shuffle=False,
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_dataloader, eval_dataloader


if __name__ == "__main__":
    
    small_pretrained_model_cpt = 'EleutherAI/pythia-410m'
    small_finetuned_model_cpt = 'models/pythia_410m_r=8_0.0001_gsm8k'
    large_pretrained_model_cpt = 'EleutherAI/pythia-1.4b'
    large_finetuned_model_cpt = 'models/pythia_1.4b_r=8_0.0001_gsm8k'


    # Load the models
    small_pretrained_model = eval_utils.load_model(small_pretrained_model_cpt)
    small_finetuned_model = eval_utils.load_model(small_pretrained_model_cpt, small_finetuned_model_cpt)
    large_pretrained_model = eval_utils.load_model(large_pretrained_model_cpt)
    large_finetuned_model = eval_utils.load_model(large_pretrained_model_cpt, large_finetuned_model_cpt)

    # Get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(small_pretrained_model_cpt)

    # Grow the small_pretrained_model and small_finetuned_model
    grown_pretrained_model = grow_model_410m_to_1_4b(small_pretrained_model)
    grown_finetuned_model = grow_model_410m_to_1_4b(small_finetuned_model)

    # Find difference between model parameters of grown_pretrained_model and grown_finetuned_model, 
    # store the difference in a state dict
    finetune_diff_state_dict = get_model_diff(grown_pretrained_model, grown_finetuned_model)

    # Create a shifted_large_pretrained_model by adding the model_diff_state_dict to large_pretrained_model
    shifted_large_pretrained_model = copy.deepcopy(large_pretrained_model)
    
    # Add values of model_diff_state_dict to a values of shifted_large_pretrained_model
    new_state_dict = {}
    for key, value in shifted_large_pretrained_model.state_dict().items():
        new_state_dict[key] = value + finetune_diff_state_dict[key]

    # Update the state dict of shifted_large_pretrained_model
    shifted_large_pretrained_model.load_state_dict(new_state_dict)

    # Test dataloader
    # eval_dataloader = torch.load("./data/eval_dataloader.pt")
    _, eval_dataloader = get_dataloader("gsm8k", batch_size=4)

    # Define accelerator
    accelerator = Accelerator()

    # Delete small models to clear up memory
    del small_pretrained_model, small_finetuned_model, grown_pretrained_model, grown_finetuned_model
    gc.collect()
    
    # Prepare the models for evaluation
    large_pretrained_model, shifted_large_pretrained_model, large_finetuned_model, tokenizer = accelerator.prepare(
        large_pretrained_model, 
        shifted_large_pretrained_model, 
        large_finetuned_model, 
        tokenizer
    )

    # 1. Interpolate large_pretrained_model and shifted_large_pretrained_model
    losses, alphas, rouges = Interpolate.interpolate_models_by_weights_loss_rouge(
        large_pretrained_model,
        shifted_large_pretrained_model,
        3,
        eval_dataloader,
        tokenizer,
        accelerator,
    )
    
    rouges = [r['rougeL'] for r in rouges]

    # Plot the losses
    eval_utils.plot_losses(
        alphas, 
        losses, 
        save_path="interpolation_loss_lpt_slpt.png",
        title="Model Evaluation Loss During Interpolation (Large Pretrained Model to Shifted Large Pretrained Model)",
        xlabel="Interpolation Alpha (fraction of large pretrained model)",
        ylabel="Loss"
    )

    # Plot the rouge scores
    eval_utils.plot_losses(
        alphas, 
        rouges, 
        save_path="interpolation_rouge_lpt_slpt.png",
        title="Model Evaluation Rouge Scores During Interpolation (Large Pretrained Model to Shifted Large Pretrained Model)",
        xlabel="Interpolation Alpha (fraction of large pretrained model)",
        ylabel="Rouge Score"
    )


    # 2. Interpolate shifted_large_pretrained_model and large_finetuned_model
    losses, alphas, rouges = Interpolate.interpolate_models_by_weights_loss_rouge(
        shifted_large_pretrained_model,
        large_finetuned_model,
        3,
        eval_dataloader,
        tokenizer,
        accelerator,
    )

    rouges = [r['rougeL'] for r in rouges]

    # Plot the losses
    eval_utils.plot_losses(
        alphas, 
        losses, 
        save_path="interpolation_loss_slpt_lft.png", 
        title="Model Evaluation Loss During Interpolation (Shifted Large Pretrained Model to Large Finetuned Model)",
        xlabel="Interpolation Alpha (fraction of shifted large pretrained model)",
        ylabel="Loss"
    )

    # Plot the rouge scores
    eval_utils.plot_losses(
        alphas, 
        rouges, 
        save_path="interpolation_rouge_slpt_lft.png",
        title="Model Evaluation Rouge Scores During Interpolation (Shifted Large Pretrained Model to Large Finetuned Model)",
        xlabel="Interpolation Alpha (fraction of shifted large pretrained model)",
        ylabel="RougeL high Recall Score"
    )