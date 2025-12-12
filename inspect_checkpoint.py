import torch
import os
import re

checkpoint_path = 'scripts/SVQ/exp_study/checkpoints/False_ts100_PatchDN_96_192_SVQ_weather_ftM_sl96_ll48_pl192_dm256_nh8_el2_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth'

def parse_filename(path):
    filename = os.path.basename(os.path.dirname(path))
    print(f"Analyzing checkpoint from folder: {filename}")
    
    # Regex to extract parameters
    params = {}
    
    patterns = {
        'model': r'PatchDN',
        'dataset': r'weather', 
        'seq_len': r'sl(\d+)',
        'label_len': r'll(\d+)',
        'pred_len': r'pl(\d+)',
        'd_model': r'dm(\d+)',
        'n_heads': r'nh(\d+)',
        'e_layers': r'el(\d+)',
        'd_layers': r'dl(\d+)',
        'd_ff': r'df(\d+)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, filename)
        if match:
            if key == 'model' or key == 'dataset':
                 params[key] = match.group(0) 
            else:
                params[key] = int(match.group(1))
    
    return params

def inspect_weights(checkpoint):
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and any(k.startswith('diffussion_model') for k in checkpoint.keys()):
        state_dict = checkpoint
    else:
        print("Could not identify state_dict format.")
        return

    print("\nInferred Model Architecture from Weights:")
    
    # Check for PatchDN structure
    if any(k.startswith('diffussion_model') for k in state_dict.keys()):
        print("Model Type: Diffusion Model (likely containing PatchDN)")
        
        # Count blocks
        block_keys = [k for k in state_dict.keys() if 'diffussion_model.blocks.' in k]
        block_indices = set()
        for k in block_keys:
            match = re.search(r'blocks\.(\d+)\.', k)
            if match:
                block_indices.add(int(match.group(1)))
        
        if block_indices:
            print(f"Number of PatchDN Blocks (depth): {max(block_indices) + 1}")
        
        # Check hidden size from x_embedder or blocks
        if 'diffussion_model.x_embedder.value_embedding.weight' in state_dict:
            weight = state_dict['diffussion_model.x_embedder.value_embedding.weight']
            # Shape is [out_features, in_features] for Linear
            print(f"x_embedder.value_embedding.weight shape: {weight.shape}")
            print(f"  -> Hidden Size (d_model_d): {weight.shape[0]}") 
            print(f"  -> Patch Length (patch_size): {weight.shape[1]}")
            
        if 'diffussion_model.blocks.0.attn.qkv.weight' in state_dict:
             weight = state_dict['diffussion_model.blocks.0.attn.qkv.weight']
             print(f"Attention QKV weight shape: {weight.shape}")
             # Shape is [3 * hidden_size, hidden_size]
             hidden_size = weight.shape[1]
             print(f"  -> Hidden Size (confirmed): {hidden_size}")

if not os.path.exists(checkpoint_path):
    print(f"Error: File not found at {checkpoint_path}")
else:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        print("Checkpoint loaded successfully.")
        
        params = parse_filename(checkpoint_path)
        print("\nParameters parsed from filename:")
        for k, v in params.items():
            print(f"  {k}: {v}")
            
        inspect_weights(checkpoint)

    except Exception as e:
        print(f"Error: {e}")
