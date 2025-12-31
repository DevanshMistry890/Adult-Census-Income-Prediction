// REMOVE import * as ort
import { KernelSHAP } from 'webshap';

// --- ACCESS GLOBAL ORT ---
// We access the library loaded from the script tag in index.html
const ort = (window as any).ort;

// --- CONFIGURATION ---
// Tell it to use the WASM files from the same CDN version
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/";

// Uses the base path automatically (works for both localhost and GitHub Pages)
const MODEL_PATH = import.meta.env.BASE_URL + 'model.onnx';
const BACKGROUND_PATH = import.meta.env.BASE_URL + 'background.json';

let session: any = null; // Type 'any' because we don't have the TS types installed
let backgroundData: number[][] | null = null;

export const initAI = async () => {
  if (session && backgroundData) return true;

  try {
    console.log("⏳ Loading Edge AI Engine (Global Script)...");
    
    // Create Session using Global ORT
    session = await ort.InferenceSession.create(MODEL_PATH, { 
        executionProviders: ['wasm'], 
        graphOptimizationLevel: 'all',
        executionMode: 'sequential', 
    });
    
    // Load Background Data for SHAP
    const res = await fetch(BACKGROUND_PATH);
    backgroundData = await res.json();
    
    console.log("✅ Edge AI Ready: Loaded via CDN");
    return true;
  } catch (e) {
    console.error("❌ AI Init Failed:", e);
    return false;
  }
};

// Wrapper for WebSHAP
const modelWrapper = async (inputs: number[][]): Promise<number[][]> => {
  if (!session) throw new Error("Session not ready");

  const flatData = inputs.flat();
  // Use Float32Array
  const tensor = new ort.Tensor('float32', Float32Array.from(flatData), [inputs.length, 12]);
  
  const results = await session.run({ float_input: tensor });

  const probabilities: number[][] = [];
  if (results.label && results.label.data) {
     for (let i = 0; i < inputs.length; i++) {
        const label = Number(results.label.data[i]);
        probabilities.push([label]); 
     }
  }
  return probabilities;
};

export const runEdgeInference = async (inputRow: number[]) => {
  if (!session || !backgroundData) await initAI();

  const probs2D = await modelWrapper([inputRow]);
  const probability = probs2D[0][0]; 
  const prediction = probability > 0.5 ? 1 : 0;

  const explainer = new KernelSHAP(modelWrapper, backgroundData!, 0.2022);
  const shapValues = await explainer.explainOneInstance(inputRow, 200);

  return {
    prediction,
    probability, 
    shap_values: shapValues 
  };

};
