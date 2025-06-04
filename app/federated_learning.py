import torch
import numpy as np
import os
import argparse

# --- CONFIGURATION ---
# List your checkpoint files and their metadata here
CHECKPOINTS = [
    {
        'path': 'model1.ckpt',
        'epochs': 5,
        'files': 100
    },
    {
        'path': 'model2.ckpt',
        'epochs': 10,
        'files': 200
    }
]
# Output path for aggregated checkpoint\OUTPUT_PATH = 'aggregated_model.ckpt'

# Function to recreate your model
# Replace with your actual model constructor
from my_model import create_model


def load_state_dict_from_ckpt(ckpt_path):
    """
    Loads a checkpoint and returns its state_dict (OrderedDict).
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # if checkpoint is a dict with 'state_dict', extract it
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        return ckpt['state_dict']
    # if it's already a state_dict
    return ckpt


def extract_weights_list(state_dict):
    """
    Converts OrderedDict of tensors into a list of numpy arrays.
    Maintains key ordering.
    """
    weights = []
    keys = []
    for k, v in state_dict.items():
        weights.append(v.cpu().numpy())
        keys.append(k)
    return keys, weights


def weighted_average(weights_list, files_list):
    """
    Weighted average of list of weight-lists by files_list.
    weights_list: list of lists of numpy arrays
    files_list: list of ints
    Returns aggregated list of numpy arrays.
    """
    total_files = sum(files_list)
    if total_files == 0:
        raise ValueError("Total files count is zero.")

    num_models = len(weights_list)
    num_layers = len(weights_list[0])
    # check consistency
    for w in weights_list:
        if len(w) != num_layers:
            raise ValueError("Inconsistent number of layers among models.")

    aggregated = []
    for layer_idx in range(num_layers):
        layer_sum = np.zeros_like(weights_list[0][layer_idx])
        for i in range(num_models):
            layer_sum += (files_list[i] / total_files) * weights_list[i][layer_idx]
        aggregated.append(layer_sum)
    return aggregated


def rebuild_state_dict(keys, weights_arrays):
    """
    Reconstruct an OrderedDict state_dict from keys and numpy arrays.
    """
    new_state = {}
    for k, arr in zip(keys, weights_arrays):
        new_state[k] = torch.tensor(arr)
    return new_state


def main():
    # Prepare lists
    all_keys = None
    all_weights = []
    file_counts = []

    # Load each checkpoint
    for entry in CHECKPOINTS:
        path = entry['path']
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint '{path}' not found.")

        state_dict = load_state_dict_from_ckpt(path)
        keys, weights = extract_weights_list(state_dict)

        if all_keys is None:
            all_keys = keys
        elif keys != all_keys:
            raise ValueError("State_dict parameter keys do not match across checkpoints.")

        all_weights.append(weights)
        file_counts.append(entry['files'])
        print(f"Loaded {path}: files={entry['files']}")

    # Aggregate weights
    aggregated_weights = weighted_average(all_weights, file_counts)
    print("Aggregated weights computed.")

    # Rebuild state_dict and save
    new_state = rebuild_state_dict(all_keys, aggregated_weights)
    # Optionally include metadata
    save_dict = {
        'state_dict': new_state,
        'merged_files': sum(file_counts),
        'num_models': len(CHECKPOINTS)
    }
    torch.save(save_dict, OUTPUT_PATH)
    print(f"Saved aggregated checkpoint to '{OUTPUT_PATH}'")


if __name__ == '__main__':
    main()
