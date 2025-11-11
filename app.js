/* Minimal ONNXRuntime-Web inference with optional meta.json support.
   - If meta.json exists: uses feature_names and optional scaler stats (z-score).
   - If not: falls back to manual constants (no scaling).
*/

// ======= CONFIGURE THESE =======
const MODEL_URL    = "model.onnx";     // Change to your .onnx path (e.g., "artifacts/your_model.onnx")
const INPUT_NAME   = "input";          // Change if your model's input name differs (e.g., "X")
const OUTPUT_NAME  = "output";         // Change if your model's output name differs (e.g., "yhat")

// If you DO NOT have meta.json, define feature names (preferred) OR just the count:
const FEATURE_NAMES = [];              // e.g., ["f1","f2","f3","f4"]  (leave [] if using FEATURE_COUNT)
const FEATURE_COUNT = 0;               // e.g., 10  (ignored if FEATURE_NAMES is non-empty)
// =================================

let meta = null;       // will contain feature_names and optional scaler stats if meta.json found
let session = null;    // onnxruntime-web session
let featureNames = []; // active feature names used by UI
let scaler = null;     // { mean: [], scale: [] } if available

function numInput({ id, label, value = 0 }) {
  const wrap = document.createElement("div");
  wrap.style.marginBottom = "10px";
  wrap.innerHTML =
    `<label for="${id}">${label}</label>
     <input type="number" id="${id}" step="any" value="${value}">`;
  return wrap;
}

async function tryLoadMeta() {
  // Try meta.json in same folder as model, then root
  const candidates = [
    MODEL_URL.replace(/[^/]+$/, "meta.json"), // same dir as model
    "meta.json"                               // project root
  ];
  for (const url of candidates) {
    try {
      const r = await fetch(url);
      if (r.ok) return await r.json();
    } catch (_) { /* ignore */ }
  }
  return null;
}

async function init() {
  const status = document.getElementById("status");
  status.textContent = "Loading configuration…";

  // Try to load meta.json (optional)
  meta = await tryLoadMeta();

  // Determine feature names
  if (meta && Array.isArray(meta.feature_names) && meta.feature_names.length > 0) {
    featureNames = meta.feature_names.slice();
  } else if (Array.isArray(FEATURE_NAMES) && FEATURE_NAMES.length > 0) {
    featureNames = FEATURE_NAMES.slice();
  } else if (Number.isInteger(FEATURE_COUNT) && FEATURE_COUNT > 0) {
    featureNames = Array.from({ length: FEATURE_COUNT }, (_, i) => `f${i+1}`);
  } else {
    // Last resort: one free-text row only
    featureNames = [];
  }

  // Optional scaler from meta.json
  if (meta && Array.isArray(meta.scaler_mean) && Array.isArray(meta.scaler_scale)) {
    scaler = {
      mean: meta.scaler_mean.map(Number),
      scale: meta.scaler_scale.map(Number),
    };
  }

  // Build inputs UI
  const inputsDiv = document.getElementById("inputs");
  inputsDiv.innerHTML = "";
  if (featureNames.length > 0) {
    featureNames.forEach((name) => {
      const id = `f_${name}`;
      inputsDiv.appendChild(numInput({ id, label: name, value: 0 }));
    });
  } else {
    const info = document.createElement("div");
    info.className = "kv";
    info.textContent =
      "No meta.json and no FEATURE_NAMES/FEATURE_COUNT set.\n" +
      "Use the Quick Paste box to provide a single comma-separated row for inference.";
    inputsDiv.appendChild(info);
  }

  // Show meta/config info
  const modelPart = `Model file: ${MODEL_URL}`;
  const ioPart = `Input name: ${INPUT_NAME}\nOutput name: ${OUTPUT_NAME}`;
  const featsPart = (featureNames.length > 0)
      ? `Features (${featureNames.length}): ${featureNames.join(", ")}`
      : "Features: (using Quick Paste only)";
  const scalePart = (scaler)
      ? "Scaling: z-score (mean/scale from meta.json)"
      : "Scaling: none (raw values passed to model)";
  const namePart = (meta && meta.best_model_name) ? `Model label: ${meta.best_model_name}` : "";

  document.getElementById("metaInfo").textContent = [modelPart, ioPart, featsPart, scalePart, namePart]
    .filter(Boolean)
    .join("\n");

  // Load ONNX model
  status.textContent = "Loading ONNX model…";
  session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
  status.textContent = "Ready.";

  document.getElementById("predictBtn").addEventListener("click", predict);
  document.getElementById("resetBtn").addEventListener("click", resetUI);
}

function resetUI() {
  if (featureNames.length > 0) {
    featureNames.forEach((name) => {
      document.getElementById(`f_${name}`).value = 0;
    });
  }
  document.getElementById("csvRow").value = "";
  document.getElementById("result").textContent = "—";
  document.getElementById("status").textContent = "Ready.";
}

function zscoreRow(values, mean, scale) {
  return values.map((v, i) => (v - mean[i]) / scale[i]);
}

function getRowValues() {
  // If Quick Paste is filled, use it
  const csv = (document.getElementById("csvRow").value || "").trim();
  if (csv.length > 0) {
    const arr = csv.split(",").map(s => parseFloat(s.trim())).filter(v => !Number.isNaN(v));
    return arr;
  }

  // Otherwise read from individual inputs
  if (featureNames.length > 0) {
    return featureNames.map(n => parseFloat(document.getElementById(`f_${n}`).value || "0"));
  }

  // Nothing to read
  return [];
}

async function predict() {
  try {
    if (!session) return;
    const status = document.getElementById("status");
    status.textContent = "Running…";

    let row = getRowValues();
    if (row.length === 0) throw new Error("No input values provided.");

    // If we have featureNames, ensure length matches
    if (featureNames.length > 0 && row.length !== featureNames.length) {
      throw new Error(`Expected ${featureNames.length} features, got ${row.length}.`);
    }

    // Apply z-score if scaler present
    if (scaler) {
      if (scaler.mean.length !== row.length || scaler.scale.length !== row.length) {
        throw new Error("Scaler length does not match feature count.");
      }
      row = zscoreRow(row, scaler.mean, scaler.scale);
    }

    // Build tensor [1, n_features]
    const inputTensor = new ort.Tensor("float32", Float32Array.from(row), [1, row.length]);

    // Run inference
    const outputs = await session.run({ [INPUT_NAME]: inputTensor });
    const out = outputs[OUTPUT_NAME];

    let predText = "";
    if (!out) {
      throw new Error(`Output named "${OUTPUT_NAME}" not found. Check OUTPUT_NAME.`);
    }
    if (Array.isArray(out.data)) {
      predText = Array.from(out.data).map(v => Number(v).toFixed(6)).join(", ");
    } else {
      predText = Array.from(out.data).map(v => Number(v).toFixed(6)).join(", ");
    }

    document.getElementById("result").textContent = predText || "—";
    status.textContent = "Done.";
  } catch (err) {
    console.error(err);
    document.getElementById("status").textContent = `Error: ${err.message}`;
  }
}

document.addEventListener("DOMContentLoaded", init);
