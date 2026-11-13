import React, { useRef, useState, useEffect, useCallback } from "react";
import { api } from "../services/api.js";
import { Upload, Loader2, AlertCircle, Eye } from "lucide-react";

// ---------------------------------------------------------------------------
// Canvas overlay: draws nodule bounding boxes and masks on CT image
// ---------------------------------------------------------------------------
function NoduleCanvas({ imageUrl, nodules }) {
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

      nodules.forEach((n, i) => {
        const [x1, y1, x2, y2] = n.box;
        const hue  = (i * 60) % 360;
        const col  = `hsl(${hue}, 90%, 60%)`;
        // Box
        ctx.strokeStyle = col;
        ctx.lineWidth   = 1.5;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        // Label
        const label = `${(n.score * 100).toFixed(0)}% · ${n.diameter_estimate_mm?.toFixed(1) ?? "?"}mm`;
        ctx.fillStyle    = col;
        ctx.font         = "9px monospace";
        ctx.fillRect(x1, y1 - 13, label.length * 5.5, 13);
        ctx.fillStyle    = "#000";
        ctx.fillText(label, x1 + 2, y1 - 3);
      });
    };
    img.src = imageUrl;
  }, [imageUrl, nodules]);

  return (
    <canvas
      ref={canvasRef}
      className="max-w-full rounded-lg border border-slate-700"
      style={{ imageRendering: "pixelated", maxHeight: 480 }}
    />
  );
}

// ---------------------------------------------------------------------------
// Main Inference Page
// ---------------------------------------------------------------------------
export default function InferencePage() {
  const [file, setFile]       = useState(null);
  const [imageUrl, setUrl]    = useState(null);
  const [scanUid, setScanUid] = useState("");
  const [expId, setExpId]     = useState("");
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const [history, setHistory] = useState([]);
  const [experiments, setExps]= useState([]);
  const fileRef               = useRef(null);

  useEffect(() => {
    api.listExperiments({ status: "done" }).then(setExps).catch(() => {});
    api.listInferences({ limit: 10 }).then(setHistory).catch(() => {});
  }, []);

  const onFileChange = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setUrl(URL.createObjectURL(f));
    setResult(null);
    if (!scanUid) setScanUid(`scan_${Date.now()}`);
  };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f) { setFile(f); setUrl(URL.createObjectURL(f)); setResult(null); }
  }, []);

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

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-bold text-white">Nodule Inference</h2>
        <p className="text-sm text-slate-400">Upload a 2.5D CT patch — Mask R-CNN detects and segments nodules</p>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
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
              {file ? file.name : "Drop a CT patch PNG/JPEG here\nor click to browse"}
            </div>
            {file && <div className="text-xs text-slate-500">{(file.size / 1024).toFixed(1)} KB</div>}
            <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={onFileChange} />
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
            onClick={runInference} disabled={loading || !file}
          >
            {loading ? <Loader2 size={16} className="animate-spin" /> : <Eye size={16} />}
            {loading ? "Running Inference..." : "Run Mask R-CNN"}
          </button>
        </div>

        {/* Results panel */}
        <div className="space-y-4">
          {/* Canvas viewer */}
          <div className="card flex items-center justify-center min-h-64 bg-slate-950">
            {imageUrl ? (
              <NoduleCanvas imageUrl={imageUrl} nodules={result?.nodules ?? []} />
            ) : (
              <div className="text-slate-600 text-sm text-center">
                <div className="text-4xl mb-2">🫁</div>
                Upload a CT patch to see predictions
              </div>
            )}
          </div>

          {/* Detection table */}
          {result && (
            <div className="card">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-bold text-white">
                  {result.nodules.length} Nodule{result.nodules.length !== 1 ? "s" : ""} Detected
                </h3>
                <span className="text-xs text-slate-400">{result.inference_time_ms?.toFixed(1)} ms</span>
              </div>
              {result.nodules.length === 0 ? (
                <p className="text-xs text-slate-500">No nodules found above the detection threshold.</p>
              ) : (
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
              )}
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
                {["ID","Scan UID","Nodules","Dice","Source","Time (ms)","At"].map(h => (
                  <th key={h} className="text-left py-2 px-3 text-slate-400 uppercase tracking-wider">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {history.map((r) => (
                <tr key={r.id} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                  <td className="py-2 px-3 text-slate-400">#{r.id}</td>
                  <td className="py-2 px-3 font-mono text-cyan-400">{r.scan_uid}</td>
                  <td className="py-2 px-3">{r.nodules?.length ?? 0}</td>
                  <td className="py-2 px-3 text-green-400">{r.dice?.toFixed(4) ?? "—"}</td>
                  <td className="py-2 px-3">
                    <span className={`badge ${r.source === "edge" ? "badge-purple" : "badge-cyan"}`}>{r.source}</span>
                  </td>
                  <td className="py-2 px-3 text-slate-400">{r.inference_time_ms?.toFixed(1) ?? "—"}</td>
                  <td className="py-2 px-3 text-slate-500">{new Date(r.created_at).toLocaleTimeString()}</td>
                </tr>
              ))}
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
