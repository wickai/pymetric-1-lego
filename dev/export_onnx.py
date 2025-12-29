import torch
import json
import os
import sys
from nas_common import MobileNetSearchSpace

def main():
    # 1. Load Architecture from test_arch.json
    arch_path = "test_arch.json"
    if not os.path.exists(arch_path):
        print(f"Error: {arch_path} not found.")
        sys.exit(1)
        
    with open(arch_path, "r") as f:
        arch_data = json.load(f)
    
    op_codes = arch_data.get('op_codes')
    width_codes = arch_data.get('width_codes')
    
    print(f"Loaded architecture from {arch_path}")
    print(f"op_codes len: {len(op_codes)}")
    
    # Define model path
    model_path = "train_cifar_4card.pth"
    if not os.path.exists(model_path):
        # Fallback to logs directory as seen in user logs
        if os.path.exists("logs/train_cifar_4card.pth"):
            model_path = "logs/train_cifar_4card.pth"
        else:
            print(f"Error: {model_path} not found.")
            # Don't exit here, let torch.load try or fail, or maybe we should exit?
            # User might have the file elsewhere. Let's just warn or let it fail in torch.load if strict.
            # But let's check properly.
            if len(sys.argv) > 1:
                model_path = sys.argv[1]
                print(f"Using model path from arg: {model_path}")
    
    print(f"Loading weights from {model_path}...")

    try:
        # Set weights_only=False to allow loading arbitrary objects (like the whole model)
        # This is required for checkpoints that save the whole model object (pickle)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        # Fallback for older pytorch versions that don't support weights_only
        checkpoint = torch.load(model_path, map_location="cpu")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)
    
    model = None
    
    if isinstance(checkpoint, torch.nn.Module):
        print("Checkpoint is a full model object. Using it directly.")
        model = checkpoint
    else:
        print("Checkpoint is a state_dict or dict. Reconstructing model from JSON...")
        
        # Reconstruct Model
        # Assuming CIFAR-10 defaults as per train_cifar.py (num_classes=10, small_input=True)
        sp = MobileNetSearchSpace(num_classes=10, small_input=True)
        
        # Check if we should use get_model or get_prefix_model
        if len(op_codes) < sp.total_blocks:
            print(f"Building PREFIX model with {len(op_codes)} blocks (total {sp.total_blocks})")
            model = sp.get_prefix_model(op_codes, width_codes)
        else:
            print(f"Building FULL model with {len(op_codes)} blocks")
            model = sp.get_model(op_codes, width_codes)

        state_dict = None
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            print("Checkpoint is a dict with 'state_dict' key.")
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            # Check if it looks like a state dict (keys are strings)
            print("Assuming checkpoint is state_dict.")
            state_dict = checkpoint
        else:
            print(f"Unknown checkpoint format: {type(checkpoint)}")
            sys.exit(1)
            
        # Handle 'module.' prefix if saved with DDP
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        try:
            model.load_state_dict(new_state_dict)
            print("Model state_dict loaded successfully.")
        except RuntimeError as e:
            print(f"Error loading state_dict: {e}")
            print("Attempting strict=False...")
            model.load_state_dict(new_state_dict, strict=False)
            print("Model loaded with strict=False.")

    model.eval()
    
    # 4. Export to ONNX
    output_dir = "output_models"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "train_cifar_4card.onnx")
    
    # Dummy input for CIFAR-10: Batch size 1, 3 channels, 32x32 image
    dummy_input = torch.randn(1, 3, 32, 32)
    input_names = ["input"]
    output_names = ["output"]
    
    print(f"Exporting to {output_path}...")
    try:
        # Using opset_version=18 because newer PyTorch (2.6+) defaults to this and 
        # the automatic version converter from 18 -> 12/13 is currently failing.
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=18,
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=True
        )
        print(f"Export complete: {output_path}")
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
