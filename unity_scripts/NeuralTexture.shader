Shader "Unlit/NeuralTexture"
{
    Properties
    {
        _Grid0 ("Grid 0 (RGBA)", 2D) = "gray" {}
        _Grid1 ("Grid 1 (RGBA)", 2D) = "gray" {}
        _Grid2 ("Grid 2 (RGBA)", 2D) = "gray" {}
        _Grid3 ("Grid 3 (RGBA)", 2D) = "gray" {}
        _Grid4 ("Grid 4 (RGBA)", 2D) = "gray" {}
        _Grid5 ("Grid 5 (RGBA)", 2D) = "gray" {}
        _Grid6 ("Grid 6 (RGBA)", 2D) = "gray" {}
        _Grid7 ("Grid 7 (RGBA)", 2D) = "gray" {}
        
        _Levels ("Pyramid Levels", Int) = 6
        _FeatureDim ("Feature Dim", Int) = 4
        
        _GlobalScale ("Brightness Scale", Range(0.1, 5.0)) = 1.0
        [Toggle] _FlipY ("Flip Y", Float) = 0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        LOD 100
        
        ZWrite On
        ZTest LEqual

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 5.0

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            // --- VARIABLES ---
            sampler2D _Grid0; sampler2D _Grid1;
            sampler2D _Grid2; sampler2D _Grid3;
            sampler2D _Grid4; sampler2D _Grid5;
            sampler2D _Grid6; sampler2D _Grid7;
            
            int _Levels;
            int _FeatureDim;
            half _GlobalScale;
            half _FlipY;

            // Buffers (Read as float, used as half)
            StructuredBuffer<float> _W1; StructuredBuffer<float> _B1;
            StructuredBuffer<float> _W2; StructuredBuffer<float> _B2;
            StructuredBuffer<float> _W3; StructuredBuffer<float> _B3;

            static const int HIDDEN_DIM = 64;
            // Max input dim = 8 levels * 4 features = 32
            // If feature dim > 4, this shader needs update
            static const int MAX_INPUT_DIM = 32; 

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            inline half relu(half x) { return max((half)0.0, x); }
            inline half sigmoid(half x) { return (half)1.0 / ((half)1.0 + exp(-x)); }

            // Helper to read 4 floats and cast to half4
            inline half4 read4(StructuredBuffer<float> buf, int offset) {
                return half4((half)buf[offset], (half)buf[offset+1], (half)buf[offset+2], (half)buf[offset+3]);
            }

            [earlydepthstencil]
            fixed4 frag (v2f i) : SV_Target
            {
                half2 uv = i.uv;
                if (_FlipY > 0.5) uv.y = 1.0 - uv.y;

                // 1. FEATURE EXTRACTION (Pyramid)
                // ---------------------
                // We sample all activated levels.
                // Assuming FeatureDim = 4 for now (RGBA)
                
                half4 features[8]; // Max 8 levels
                
                // Unroll or branch? 
                // Branching on _Levels might work
                
                if (_Levels > 0) features[0] = tex2D(_Grid0, uv);
                if (_Levels > 1) features[1] = tex2D(_Grid1, uv);
                if (_Levels > 2) features[2] = tex2D(_Grid2, uv);
                if (_Levels > 3) features[3] = tex2D(_Grid3, uv);
                if (_Levels > 4) features[4] = tex2D(_Grid4, uv);
                if (_Levels > 5) features[5] = tex2D(_Grid5, uv);
                if (_Levels > 6) features[6] = tex2D(_Grid6, uv);
                if (_Levels > 7) features[7] = tex2D(_Grid7, uv);
                
                // Flatten features into input vector for MLP
                // InputDim = _Levels * 4
                
                // 2. MLP INFERENCE
                // ----------------
                // Layer 1 [InputDim -> 64]
                half l1[HIDDEN_DIM];
                
                [unroll(HIDDEN_DIM)]
                for(int h=0; h<HIDDEN_DIM; h++)
                {
                    half sum = (half)_B1[h];
                    int w_off = h * (_Levels * 4); // Stride depends on input dim!
                    
                    // Manually accumulate dot products based on active levels
                    // We can loop over levels
                    
                    [unroll(8)] // Max levels
                    for(int l=0; l<8; l++)
                    {
                        if (l < _Levels)
                        {
                            // Each feature is half4
                            // Weight chunk is 4 floats
                            sum += dot(features[l], read4(_W1, w_off + l * 4));
                        }
                    }

                    l1[h] = relu(sum);
                }

                // Layer 2 [64 -> 64]
                half l2[HIDDEN_DIM];
                
                [unroll(HIDDEN_DIM)]
                for(int h=0; h<HIDDEN_DIM; h++)
                {
                    half sum = (half)_B2[h];
                    int w_off = h * 64;

                    // Loop unroll for 16 * half4 = 64
                    [unroll]
                    for(int j=0; j<64; j+=4)
                    {
                         half4 prev = half4(l1[j], l1[j+1], l1[j+2], l1[j+3]);
                         sum += dot(prev, read4(_W2, w_off + j));
                    }
                    l2[h] = relu(sum);
                }

                // Layer 3 [64 -> 3]
                half3 rgb;
                
                [unroll(3)]
                for(int c=0; c<3; c++)
                {
                    half sum = (half)_B3[c];
                    int w_off = c * 64;

                    [unroll]
                    for(int j=0; j<64; j+=4)
                    {
                         half4 prev = half4(l2[j], l2[j+1], l2[j+2], l2[j+3]);
                         sum += dot(prev, read4(_W3, w_off + j));
                    }
                    rgb[c] = sigmoid(sum);
                }

                rgb *= _GlobalScale;
                return fixed4(saturate(rgb), 1.0);
            }
            ENDCG
        }
    }
}
