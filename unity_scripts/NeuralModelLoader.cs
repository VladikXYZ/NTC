using UnityEngine;
using System.IO;
using System.Collections.Generic;
using System.Linq;

[System.Serializable]
public class LayerData
{
    public string name;
    public int[] shape;
    public float[] data;
}

[System.Serializable]
public class ModelMetadata
{
    public string model_type;
    public int levels;
    public int feature_dim;
}

[System.Serializable]
public class ModelData
{
    public ModelMetadata metadata;
    public LayerData[] layers;
}

public class NeuralModelLoader : MonoBehaviour
{
    [Header("Configuration")]
    public string jsonFileName = "model_weights.json";
    public Material targetMaterial;

    [Header("Debug")]
    public bool loadOnStart = true;

    // Keep references to buffers to release them
    private List<ComputeBuffer> activeBuffers = new List<ComputeBuffer>();

    void Start()
    {
        if (loadOnStart) LoadModel();
    }

    [ContextMenu("Load Model")]
    public void LoadModel()
    {
        string path = Path.Combine(Application.dataPath, jsonFileName);
        if (!File.Exists(path))
        {
            // Try looking in project root (for editor convenience)
            path = Path.Combine(Directory.GetParent(Application.dataPath).FullName, jsonFileName);
            if (!File.Exists(path))
            {
                Debug.LogError($"Could not find JSON file at: {path}");
                return;
            }
        }

        Debug.Log($"Loading model from: {path}");
        string json = File.ReadAllText(path);
        
        // Parse JSON
        ModelData modelData = JsonUtility.FromJson<ModelData>(json);
        if (modelData == null || modelData.layers == null)
        {
            Debug.LogError("Failed to parse model data. Ensure the JSON format is correct.");
            return;
        }

        // Clear old buffers
        ReleaseBuffers();

        // Pass Metadata to Shader
        if (modelData.metadata != null)
        {
            targetMaterial.SetInt("_Levels", modelData.metadata.levels);
            targetMaterial.SetInt("_FeatureDim", modelData.metadata.feature_dim);
            Debug.Log($"Model Metadata: Levels={modelData.metadata.levels}, Feats={modelData.metadata.feature_dim}");
        }

        // Process Layers
        foreach (var layer in modelData.layers)
        {
            ProcessLayer(layer);
        }
        
        Debug.Log("✓ Model Loaded Successfully!");
    }

    void ProcessLayer(LayerData layer)
    {
        string name = layer.name;
        
        // 1. Grids (grids.0, grids.1, ...)
        if (name.StartsWith("grids."))
        {
            // Extract index from "grids.X"
            string[] parts = name.Split('.');
            if (parts.Length >= 2 && int.TryParse(parts[1], out int index))
            {
                // Shape: [1, FeatureDim, H, W]
                int features = layer.shape[1]; 
                int height = layer.shape[2];
                int width = layer.shape[3]; # 
                
                // Texture: H x W, RGBAFloat (4 channels)
                // If features > 4, we might need multiple textures per level, 
                // but for now let's assume FeatureDim=4 (RGBA)
                
                if (features != 4)
                {
                    Debug.LogWarning($"Grid {index} has {features} features. Shader expects 4 (RGBA). Truncating or padding might happen.");
                }

                Texture2D tex = new Texture2D(width, height, TextureFormat.RGBAFloat, false);
                tex.filterMode = FilterMode.Bilinear;
                tex.wrapMode = TextureWrapMode.Clamp;

                Color[] colors = new Color[width * height];
                int hw = height * width;

                // Flattened data is N, C, H, W
                // data[channel * hw + pixel_idx]
                
                for (int i = 0; i < hw; i++)
                {
                    float r = layer.data[0 * hw + i];
                    float g = layer.data[1 * hw + i];
                    float b = layer.data[2 * hw + i];
                    float a = layer.data[3 * hw + i];
                    colors[i] = new Color(r, g, b, a);
                }

                tex.SetPixels(colors);
                tex.Apply();

                // Set to shader property _Grid0, _Grid1, etc.
                targetMaterial.SetTexture($"_Grid{index}", tex);
                Debug.Log($"Uploaded Grid {index} ({width}x{height})");
            }
        }
        // 2. Weights (ComputeBuffers)
        else if (name.Contains("weight") || name.Contains("bias"))
        {
            // Mapping:
            // mlp.0.weight -> _W1
            // mlp.0.bias   -> _B1
            // mlp.2.weight -> _W2
            // mlp.2.bias   -> _B2
            // mlp.4.weight -> _W3
            // mlp.4.bias   -> _B3
            
            string shaderProp = "";
            if (name == "mlp.0.weight") shaderProp = "_W1";
            if (name == "mlp.0.bias")   shaderProp = "_B1";
            if (name == "mlp.2.weight") shaderProp = "_W2";
            if (name == "mlp.2.bias")   shaderProp = "_B2";
            if (name == "mlp.4.weight") shaderProp = "_W3";
            if (name == "mlp.4.bias")   shaderProp = "_B3";

            if (shaderProp != "")
            {
                // Weights are usually [Out, In]
                // Biases are [Out]
                // We pass them flat
                ComputeBuffer buffer = new ComputeBuffer(layer.data.Length, sizeof(float));
                buffer.SetData(layer.data);
                activeBuffers.Add(buffer);
                targetMaterial.SetBuffer(shaderProp, buffer);
                Debug.Log($"Uploaded {name} to {shaderProp} ({layer.data.Length} floats)");
            }
        }
    }

    void OnDestroy()
    {
        ReleaseBuffers();
    }

    void ReleaseBuffers()
    {
        foreach (var buff in activeBuffers) buff?.Release();
        activeBuffers.Clear();
    }
}
