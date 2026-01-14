import json
import torch
import os
import numpy as np

def export_model_for_unity(model, output_name, input_sample=None, onnx=False):
    """
    Exports the PyTorch model to a JSON file containing all weights (state_dict) 
    for manual loading in Unity.
    """
    
    # Ensure output directory exists
    export_dir = "unity_exports"
    os.makedirs(export_dir, exist_ok=True)
    
    base_path = os.path.join(export_dir, output_name)
    
    # --- 1. JSON Export (Architecture Agnostic) ---
    print(f"Exporting weights to JSON for custom shaders...")
    
    # Try to extract model config if available
    metadata = {
        "model_type": type(model).__name__,
        "generated_at": str(np.datetime64('now')),
    }

    if hasattr(model, 'levels'):
        metadata['levels'] = model.levels
    if hasattr(model, 'feature_dim'):
        metadata['feature_dim'] = model.feature_dim
    
    model_data = {
        "metadata": metadata,
        "layers": []
    }
    
    state_dict = model.state_dict()
    for key, value in state_dict.items():
        # Flatten tensors and convert to list for JSON serialization
        # precision: keep as float (32-bit usually)
        weight_list = value.cpu().numpy().flatten().tolist()
        shape = list(value.shape)
        
        layer_obj = {
            "name": key,
            "shape": shape,
            "data": weight_list
        }
        model_data["layers"].append(layer_obj)
        
    json_path = f"{base_path}_weights.json"
    with open(json_path, 'w') as f:
        json.dump(model_data, f) 
    print(f"✓ Saved JSON weights to: {json_path}")
    
    # --- 2. ONNX Export (Optional/Standard) ---
    if input_sample is not None and onnx:
        try:
            onnx_path = f"{base_path}.onnx"
            torch.onnx.export(model, input_sample, onnx_path, 
                            opset_version=11, 
                            input_names=['input_uv'], 
                            output_names=['output_rgb'],
                            dynamic_axes={'input_uv': {0: 'batch_size'}, 'output_rgb': {0: 'batch_size'}})
            print(f"✓ Saved ONNX model to: {onnx_path}")
        except Exception as e:
            print(f"x Failed to export ONNX: {e}")
    
    # --- 3. C# Helper Hint ---
    print("\n--- Unity Integration Hint ---")
    print(f"For '{type(model).__name__}', parse '{json_path}' in C# using JsonUtility.")
    print("If using a custom Shader, pass these weights as ComputeBuffer or Arrays.")
    print("------------------------------\n")
