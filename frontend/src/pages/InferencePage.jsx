import React, { useRef, useState, useEffect, useCallback } from "react";
import { api } from "../services/api.js";
import { Upload, Loader2, AlertCircle, Eye, CheckCircle, XCircle, ZoomIn, ZoomOut, RotateCcw } from "lucide-react";

// Model always works on 64×64 patches; box coords are in that space.
const MODEL_PATCH_SIZE = 64;

// ---------------------------------------------------------------------------
// Canvas overlay: draws nodule bounding boxes and masks on CT image
// ---------------------------------------------------------------------------
function NoduleCanvas({ imageUrl, nodules, inferenceRan, zoom }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!imageUrl || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx    = canvas.getContext("2d");
    const img    = new Image();
    img.onload = () => {
      canvas.width  = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.drawImage(img, 0, 0);

      // Scale factor: model outputs coords in 64×64 space → displayed image space
      const scaleX = img.naturalWidth  / MODEL_PATCH_SIZE;
      const scaleY = img.naturalHeight / MODEL_PATCH_SIZE;

      if (nodules.length === 0 && inferenceRan) {
        // Show a subtle "no nodules" overlay
        ctx.fillStyle = "rgba(0,0,0,0.45)";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#4ade80";
        ctx.font = `bold ${Math.max(10, canvas.height / 8)}px sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("✓ No nodules detected", canvas.width / 2, canvas.height / 2);
        ctx.textAlign = "left";
        ctx.textBaseline = "alphabetic";
        return;
      }

      nodules.forEach((n, i) => {
        if (n.mask_rle) {
          const rgba = new Uint8ClampedArray(MODEL_PATCH_SIZE * MODEL_PATCH_SIZE * 4);
          let offset = 0;
          for (const run of n.mask_rle.split(",")) {
            const [valueText, countText] = run.split(":");
            const value = Number(valueText);
            const count = Number(countText);
            if (!Number.isInteger(count) || count < 0 || offset + count > MODEL_PATCH_SIZE ** 2) break;
            if (value === 1) {
              for (let pixel = offset; pixel < offset + count; pixel += 1) {
                const idx = pixel * 4;
                rgba[idx] = 34;
                rgba[idx + 1] = 211;
                rgba[idx + 2] = 238;
                rgba[idx + 3] = 120;
              }
            }
            offset += count;
          }
          const maskCanvas = document.createElement("canvas");
          maskCanvas.width = MODEL_PATCH_SIZE;
          maskCanvas.height = MODEL_PATCH_SIZE;
          maskCanvas.getContext("2d").putImageData(
            new ImageData(rgba, MODEL_PATCH_SIZE, MODEL_PATCH_SIZE), 0, 0
          );
          ctx.drawImage(maskCanvas, 0, 0, canvas.width, canvas.height);
        }

        const [x1raw, y1raw, x2raw, y2raw] = n.box;
        // Scale from model patch space to actual image pixels
        const x1 = x1raw * scaleX;
        const y1 = y1raw * scaleY;
        const x2 = x2raw * scaleX;
        const y2 = y2raw * scaleY;

        const hue = (i * 60) % 360;
        const col = `hsl(${hue}, 90%, 60%)`;

        // Bounding box
        ctx.strokeStyle = col;
        ctx.lineWidth   = Math.max(1, canvas.width / 128);
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Label background + text
        const label    = `${(n.score * 100).toFixed(0)}% · ${n.diameter_estimate_mm?.toFixed(1) ?? "?"}mm`;
        const fontSize = Math.max(8, canvas.height / 10);
        ctx.font       = `${fontSize}px monospace`;
        const textW    = ctx.measureText(label).width + 4;
        const textH    = fontSize + 4;
        ctx.fillStyle  = col;
        ctx.fillRect(x1, Math.max(0, y1 - textH), textW, textH);
        ctx.fillStyle  = "#000";
        ctx.fillText(label, x1 + 2, Math.max(fontSize, y1 - 3));
      });
    };
    img.src = imageUrl;
  }, [imageUrl, nodules, inferenceRan]);

  return (
    <canvas
      ref={canvasRef}
      className="mx-auto block max-w-none rounded-lg border border-slate-700"
      style={{
        width: `${MODEL_PATCH_SIZE * zoom}px`,
        height: `${MODEL_PATCH_SIZE * zoom}px`,
        imageRendering: "pixelated",
      }}
    />
  );
}

// ---------------------------------------------------------------------------
// Main Inference Page
// ---------------------------------------------------------------------------
export default function InferencePage() {
  const [file, setFile]         = useState(null);
  const [imageUrl, setUrl]      = useState(null);
  const [scanUid, setScanUid]   = useState("");
  const [expId, setExpId]       = useState("");
  const [result, setResult]     = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);
  const [history, setHistory]   = useState([]);
  const [experiments, setExps]  = useState([]);
  const [modelStatus, setModelStatus] = useState(null);
  const [imageZoom, setImageZoom] = useState(8);
  const fileRef                 = useRef(null);

  useEffect(() => {
    api.listExperiments({ status: "done" }).then(setExps).catch(() => {});
    api.listInferences({ limit: 10 }).then(setHistory).catch(() => {});
    api.getInferenceStatus().then(setModelStatus).catch((e) => {
      setModelStatus({ model_loaded: false, model_loading_error: e.message });
    });
  }, []);

  const chooseFile = useCallback(async (f) => {
    if (!f) return;
    const extension = f.name.toLowerCase().split(".").pop();
    if (!new Set(["png", "jpg", "jpeg"]).has(extension)
        || !new Set(["image/png", "image/jpeg"]).has(f.type)) {
      setError("Choose a PNG or JPEG exported as a 64×64 RGB 2.5D patch.");
      return;
    }
    const previewUrl = URL.createObjectURL(f);
    const dimensions = await new Promise((resolve) => {
      const image = new Image();
      image.onload = () => resolve([image.naturalWidth, image.naturalHeight]);
      image.onerror = () => resolve(null);
      image.src = previewUrl;
    });
    if (!dimensions || dimensions[0] !== MODEL_PATCH_SIZE || dimensions[1] !== MODEL_PATCH_SIZE) {
      URL.revokeObjectURL(previewUrl);
      setError(`Patch must be exactly ${MODEL_PATCH_SIZE}×${MODEL_PATCH_SIZE} pixels. Whole CT images are not accepted here.`);
      return;
    }
    setError(null);
    setFile(f);
    setUrl(previewUrl);
    setResult(null);
    setScanUid((previous) => previous || `scan_${Date.now()}`);
  }, []);

  const onFileChange = (e) => { void chooseFile(e.target.files?.[0]); };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f) void chooseFile(f);
  }, [chooseFile]);

  const runInference = async () => {
    if (!file) return setError("Please select a CT patch image");
    setLoading(true); setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("scan_uid", scanUid || `scan_${Date.now()}`);
      form.append("source", "cloud");
      if (expId) form.append("experiment_id", expId);
      const data = await api.predictUpload(form);
      setResult(data);
      // Refresh history
      api.listInferences({ limit: 10 }).then(setHistory).catch(() => {});
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  // Average score across all detections (or null if none)
  const avgScore = result?.nodules?.length
    ? result.nodules.reduce((s, n) => s + n.score, 0) / result.nodules.length
    : null;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-bold text-white">Nodule Inference</h2>
        <p className="text-sm text-slate-400">
          Upload an exported 64×64 RGB 2.5D patch. R, G and B must contain axial z−1, z and z+1 slices in that order. This page does not accept full CT scans or a single grayscale slice.
        </p>
      </div>

      <div className="grid lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)] gap-6">
        {/* Upload panel */}
        <div className="space-y-4">
          {/* Drop zone */}
          <div
            className={`card border-2 border-dashed flex flex-col items-center justify-center gap-3 py-10 cursor-pointer transition-colors
              ${file ? "border-cyan-500/50 bg-cyan-500/5" : "border-slate-700 hover:border-slate-500"}`}
            onDrop={onDrop}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => fileRef.current?.click()}
          >
            <Upload size={28} className={file ? "text-cyan-400" : "text-slate-500"} />
            <div className="text-sm text-slate-400 text-center">
              {file ? file.name : "Drop a 64×64 RGB 2.5D PNG/JPEG here\nor click to browse"}
            </div>
            {file && <div className="text-xs text-slate-500">{(file.size / 1024).toFixed(1)} KB</div>}
            <input ref={fileRef} type="file" accept="image/png,image/jpeg" className="hidden" onChange={onFileChange} />
          </div>

          <div className={`card flex items-start gap-3 ${modelStatus?.model_loaded ? "text-green-400" : "text-amber-400"}`}>
            {modelStatus?.model_loaded ? <CheckCircle size={18} /> : <XCircle size={18} />}
            <div className="min-w-0">
              <div className="text-sm font-semibold">
                {modelStatus?.model_loaded ? "Inference model ready" : "Inference model unavailable"}
              </div>
              <div className="text-xs text-slate-400 break-all">
                {modelStatus?.checkpoint_path || "Checking backend checkpoint…"}
              </div>
              {modelStatus?.model_loaded ? (
                <div className="text-xs text-slate-400 mt-1">
                  {modelStatus.config_id} · {modelStatus.device} · score ≥ {modelStatus.score_threshold}
                </div>
              ) : (
                <div className="text-xs text-amber-300 mt-1">
                  {modelStatus?.model_loading_error || "Waiting for backend readiness status"}
                </div>
              )}
            </div>
          </div>

          {/* Options */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="label">Scan UID</label>
              <input className="input font-mono text-xs" value={scanUid}
                     onChange={e => setScanUid(e.target.value)} placeholder="scan_001" />
            </div>
            <div>
              <label className="label">Experiment (optional)</label>
              <select className="input" value={expId} onChange={e => setExpId(e.target.value)}>
                <option value="">— Best checkpoint —</option>
                {experiments.map(e => (
                  <option key={e.id} value={e.id}>{e.config_id}</option>
                ))}
              </select>
            </div>
          </div>

          {error && (
            <div className="flex items-center gap-2 text-sm text-red-400">
              <AlertCircle size={15} /> {error}
            </div>
          )}

          <button
            className="btn-primary w-full flex items-center justify-center gap-2"
            onClick={runInference} disabled={loading || !file || modelStatus?.model_loaded === false}
          >
            {loading ? <Loader2 size={16} className="animate-spin" /> : <Eye size={16} />}
            {loading ? "Running Inference..." : "Run Mask R-CNN"}
          </button>

          {/* Summary stats after inference */}
          {result && (
            <div className="card bg-slate-900 grid grid-cols-3 gap-3 text-center">
              <div>
                <div className="text-xl font-bold text-white">{result.nodules.length}</div>
                <div className="text-xs text-slate-400">Nodules</div>
              </div>
              <div>
                <div className="text-xl font-bold text-green-400">
                  {avgScore != null ? `${(avgScore * 100).toFixed(1)}%` : "—"}
                </div>
                <div className="text-xs text-slate-400">Avg Score</div>
              </div>
              <div>
                <div className="text-xl font-bold text-cyan-400">
                  {result.inference_time_ms?.toFixed(0) ?? "—"} ms
                </div>
                <div className="text-xs text-slate-400">Latency</div>
              </div>
            </div>
          )}
          {result && (
            <div className="text-xs text-slate-500 break-all">
              Checkpoint: {result.config_id || "default"} · {result.checkpoint_path}
            </div>
          )}
        </div>

        {/* Results panel */}
        <div className="space-y-4">
          {/* Canvas viewer */}
          <div className="card p-4 sm:p-5 bg-slate-950/70">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
              <div>
                <h3 className="text-sm font-bold text-white">Prediction preview</h3>
                <p className="text-xs text-slate-400 mt-1">Zoom to inspect the patch and detected regions</p>
              </div>
              {imageUrl && (
                <div className="flex items-center gap-2" aria-label="Image zoom controls">
                  <button
                    type="button"
                    className="btn btn-secondary p-2 rounded-lg disabled:opacity-40"
                    aria-label="Zoom out"
                    title="Zoom out"
                    disabled={imageZoom <= 1}
                    onClick={() => setImageZoom((zoom) => Math.max(1, zoom - 1))}
                  >
                    <ZoomOut size={16} />
                  </button>
                  <span className="min-w-12 text-center text-xs text-slate-300 tabular-nums">
                    {imageZoom * 100}%
                  </span>
                  <button
                    type="button"
                    className="btn btn-secondary p-2 rounded-lg disabled:opacity-40"
                    aria-label="Zoom in"
                    title="Zoom in"
                    disabled={imageZoom >= 16}
                    onClick={() => setImageZoom((zoom) => Math.min(16, zoom + 1))}
                  >
                    <ZoomIn size={16} />
                  </button>
                  <button
                    type="button"
                    className="btn btn-secondary p-2 rounded-lg"
                    aria-label="Reset zoom"
                    title="Reset zoom"
                    onClick={() => setImageZoom(8)}
                  >
                    <RotateCcw size={14} />
                  </button>
                </div>
              )}
            </div>
            {imageUrl ? (
              <div className="inference-image-viewport flex min-h-[420px] max-h-[70vh] justify-start overflow-auto rounded-xl border border-slate-700/70 bg-slate-950 p-4">
                <NoduleCanvas
                  imageUrl={imageUrl}
                  nodules={result?.nodules ?? []}
                  inferenceRan={result != null}
                  zoom={imageZoom}
                />
              </div>
            ) : (
              <div className="flex min-h-[420px] max-h-[70vh] items-center justify-center rounded-xl border border-slate-700/70 bg-slate-950 text-slate-600 text-sm text-center">
                <div>
                  <div className="text-4xl mb-2">🫁</div>
                  Upload a CT patch to see predictions
                </div>
              </div>
            )}
          </div>

          {/* Detection table */}
          {result && result.nodules.length > 0 && (
            <div className="card">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-bold text-white">
                  {result.nodules.length} Nodule{result.nodules.length !== 1 ? "s" : ""} Detected
                </h3>
                <span className="text-xs text-slate-400">{result.inference_time_ms?.toFixed(1)} ms</span>
              </div>
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-slate-800">
                    {["#","Score","Diameter (mm)","Box [x1,y1,x2,y2]"].map(h => (
                      <th key={h} className="text-left py-1.5 px-2 text-slate-400">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.nodules.map((n, i) => (
                    <tr key={i} className="border-b border-slate-800/50">
                      <td className="py-1.5 px-2 text-slate-400">{i+1}</td>
                      <td className="py-1.5 px-2 text-green-400 font-semibold">{(n.score*100).toFixed(1)}%</td>
                      <td className="py-1.5 px-2 text-cyan-400">{n.diameter_estimate_mm?.toFixed(2)} mm</td>
                      <td className="py-1.5 px-2 font-mono text-slate-300">
                        [{n.box.map(v => v.toFixed(1)).join(", ")}]
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {result && result.nodules.length === 0 && (
            <div className="card flex items-center gap-3 text-green-400">
              <CheckCircle size={18} />
              <div>
                <div className="text-sm font-semibold">No nodules detected</div>
                <div className="text-xs text-slate-400">
                  Inference completed in {result.inference_time_ms?.toFixed(1)} ms — no detections above threshold
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* History */}
      <div className="card">
        <h3 className="text-sm font-bold text-white mb-3">Recent Inferences</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-slate-800">
                {["ID","Scan UID","Nodules","Avg Score","Source","Time (ms)","At"].map(h => (
                  <th key={h} className="text-left py-2 px-3 text-slate-400 uppercase tracking-wider">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {history.map((r) => {
                const nodules = r.nodules ?? [];
                const avg = nodules.length
                  ? nodules.reduce((s, n) => s + (n.score ?? 0), 0) / nodules.length
                  : null;
                return (
                  <tr key={r.id} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                    <td className="py-2 px-3 text-slate-400">#{r.id}</td>
                    <td className="py-2 px-3 font-mono text-cyan-400">{r.scan_uid}</td>
                    <td className="py-2 px-3">{nodules.length}</td>
                    <td className="py-2 px-3 text-green-400">
                      {avg != null ? `${(avg * 100).toFixed(1)}%` : "—"}
                    </td>
                    <td className="py-2 px-3">
                      <span className={`badge ${r.source === "edge" ? "badge-purple" : "badge-cyan"}`}>{r.source}</span>
                    </td>
                    <td className="py-2 px-3 text-slate-400">{r.inference_time_ms?.toFixed(1) ?? "—"}</td>
                    <td className="py-2 px-3 text-slate-500">{new Date(r.created_at).toLocaleTimeString()}</td>
                  </tr>
                );
              })}
              {history.length === 0 && (
                <tr><td colSpan={7} className="py-6 text-center text-slate-500">No inference history yet</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
