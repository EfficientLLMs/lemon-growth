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



def clean_memory():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def grow_model_410m_to_1_4b(model_410m, random_noise=False):

    # # Depth from 6 -> 24
    # intermediate_grown = grow_depth.expand_layers(model_70m, 6, 12, expand_type='alternate')
    # intermediate_grown = grow_depth.expand_layers(intermediate_grown, 12, 24, expand_type='alternate')

    # Width from 512 -> 1024
    model_410m_to_1_4b = grow_width.expand_width(model_410m, 1024, 2048)

    if random_noise:
        # Add random noise to the model
        for param in model_410m_to_1_4b.parameters():
            param.data += torch.randn_like(param.data) * 1e-4

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
