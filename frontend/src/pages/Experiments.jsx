import React, { useEffect, useState, useCallback, useRef } from "react";
import { api, openTrainingWS } from "../services/api.js";
import { Plus, Play, Trash2, ChevronDown, ChevronRight, Loader2, AlertCircle, Terminal, RefreshCw } from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
} from "recharts";
import { StatusBadge } from "./Dashboard.jsx";

// ---------------------------------------------------------------------------
// Create Experiment Form
// ---------------------------------------------------------------------------
function CreateForm({ onCreated }) {
  const [runtime, setRuntime] = useState(null);
  const [loadingInfo, setLoadingInfo] = useState(true);
  const [form, setForm] = useState({
    config_id: "", phase: 1, training_strategy: "fixed",
    mgda_gradient_normalization: "raw", mgda_parameter_scope: "mask_head",
    alpha: 0.3, beta: 0.5, gamma: 0.5, delta: 0.0, epsilon: 0.0,
    epochs: 30, batch_size: 4, lr: 0.005,
    data_root: "", cache_dir: "", cache_selection: "default",
    train_subsets: "", validation_subset: "", test_subset: "",
  });
  const [loading, setLoading] = useState(false);
  const [err, setErr]         = useState(null);

  useEffect(() => {
    let active = true;
    api.getInfo()
      .then((info) => {
        if (!active) return;
        setRuntime(info);
        setForm((current) => ({
          ...current,
          train_subsets: current.train_subsets || (info.default_split?.train_subsets ?? []).join(", "),
          validation_subset: current.validation_subset || String(info.default_split?.validation_subset ?? ""),
        }));
      })
      .catch((error) => { if (active) setErr(`Could not load backend dataset configuration: ${error.message}`); })
      .finally(() => { if (active) setLoadingInfo(false); });
    return () => { active = false; };
  }, []);

  const field = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));
  const num   = (k) => (e) => setForm((f) => ({ ...f, [k]: parseFloat(e.target.value) || 0 }));

  const submit = async () => {
    if (!form.config_id.trim()) return setErr("Config ID is required");
    if (!runtime) return setErr("Backend dataset configuration is not available yet.");
    const trainSubsets = form.train_subsets
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean)
      .map(Number)
      .filter(Number.isInteger);
    if (!trainSubsets.length || form.validation_subset === "") {
      return setErr("Choose training and validation subsets before creating the experiment.");
    }
    if (form.cache_selection === "custom" && !form.cache_dir.trim()) {
      return setErr("Enter a backend-side cache directory or select the default cache.");
    }
    setLoading(true); setErr(null);
    try {
      const cacheMode = form.cache_selection === "raw" ? "raw" : "cached";
      await api.createExperiment({
        config_id: form.config_id,
        phase: Number(form.phase),
        training_strategy: form.training_strategy,
        mgda_gradient_normalization: form.mgda_gradient_normalization,
        mgda_parameter_scope: form.mgda_parameter_scope,
        loss_weights: form.training_strategy === "mgda"
          ? { alpha: 0.2, beta: 0.2, gamma: 0.2, delta: 0.2, epsilon: 0.2 }
          : { alpha: form.alpha, beta: form.beta, gamma: form.gamma,
              delta: form.delta, epsilon: form.epsilon },
        epochs: Number(form.epochs),
        batch_size: Number(form.batch_size),
        fold: Number(form.validation_subset),
        lr: Number(form.lr),
        cache_mode: cacheMode,
        patch_size: runtime.patch_size || 64,
        train_subsets: trainSubsets,
        validation_subset: Number(form.validation_subset),
        test_subset: form.test_subset === "" ? null : Number(form.test_subset),
        ...(form.data_root.trim() ? { data_root: form.data_root.trim() } : {}),
        ...(form.cache_selection === "custom" && form.cache_dir.trim()
          ? { cache_dir: form.cache_dir.trim() }
          : {}),
      });
      onCreated();
      setForm((f) => ({ ...f, config_id: "" }));
    } catch (e) { setErr(e.message); }
    finally { setLoading(false); }
  };

  const LW = [
    { k: "alpha",   label: "α BCE",      color: "text-sky-400" },
    { k: "beta",    label: "β Dice",     color: "text-green-400" },
    { k: "gamma",   label: "γ Boundary", color: "text-orange-400" },
    { k: "delta",   label: "δ Focal",    color: "text-red-400" },
    { k: "epsilon", label: "ε HD95",     color: "text-purple-400" },
  ];
  const defaultCacheStatus = loadingInfo
    ? "Checking backend…"
    : runtime?.cache_valid
      ? "Validated"
      : runtime?.cache_exists
        ? "Invalid"
        : runtime?.default_cache_dir
          ? "Missing"
          : "Not configured";
  const cacheStatus = form.cache_selection === "raw"
    ? "Raw-volume mode (explicit)"
    : form.cache_selection === "custom"
      ? (form.cache_dir.trim() ? "Validated on create" : "Custom path required")
      : defaultCacheStatus;
  const statusColor = form.cache_selection === "default" && runtime?.cache_valid
    ? "text-green-300"
    : "text-amber-300";
  const activeCachePath = form.cache_selection === "raw"
    ? "Raw CT volumes — no patch cache"
    : form.cache_selection === "custom"
      ? (form.cache_dir.trim() || "Enter a backend-side cache path below")
      : (runtime?.default_cache_dir || "Loading backend cache path…");

  return (
    <div className="card space-y-5">
      <h2 className="text-sm font-bold text-white">New Experiment</h2>

      <section className="rounded-xl border border-cyan-500/20 bg-slate-950/55 p-4 space-y-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="text-sm font-semibold text-white">Active dataset cache</div>
            <div className="mt-1 break-all font-mono text-xs text-cyan-200">{activeCachePath}</div>
          </div>
          <span className={`rounded-full border border-slate-700 px-3 py-1 text-xs font-semibold ${statusColor}`}>
            {cacheStatus}
          </span>
        </div>
        {form.cache_selection === "default" && runtime?.cache_summary && (
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-300">
            <span>{runtime.cache_summary.total_patches} patches</span>
            <span>{runtime.cache_summary.positive_patches} positive</span>
            <span>{runtime.cache_summary.negative_patches} negative</span>
            <span>{runtime.cache_summary.scan_uid_count} scans</span>
            <span>shape {runtime.cache_summary.patch_dimensions?.join("×")}</span>
            <span>spacing {runtime.cache_summary.spacing_mm?.join("×") || "not recorded"} mm</span>
          </div>
        )}
        {runtime?.cache_error && <p className="text-xs text-amber-300">{runtime.cache_error}</p>}
        <div>
          <label className="label">Dataset mode</label>
          <select className="input mt-1" value={form.cache_selection} onChange={field("cache_selection")}>
            <option value="default">Default validated patch cache</option>
            <option value="custom">Custom server-side patch cache</option>
            <option value="raw">Raw-volume mode (explicit; slower)</option>
          </select>
        </div>
        {form.cache_selection === "custom" && (
          <div>
            <label className="label">Custom cache directory on the backend machine</label>
            <input className="input mt-1 w-full font-mono" value={form.cache_dir}
                   onChange={field("cache_dir")} placeholder={runtime?.default_cache_dir || "/data/patches_cache_v3_smoke"} />
            <p className="mt-1 text-xs text-slate-500">This path is resolved and validated by the backend when you create the experiment.</p>
          </div>
        )}
        {form.cache_selection === "raw" && (
          <p className="text-xs text-amber-300">Raw CT loading is explicitly selected. This experiment will not use a patch cache.</p>
        )}
      </section>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <div className="col-span-2 md:col-span-1">
          <label className="label">Config ID</label>
          <input className="input font-mono" value={form.config_id}
                 onChange={field("config_id")} placeholder="e.g. G-a03-b05-g05" />
        </div>
        <div>
          <label className="label">Phase</label>
          <select className="input" value={form.phase} onChange={field("phase")}>
            {[1,2,3].map(p => <option key={p} value={p}>Phase {p}</option>)}
          </select>
        </div>
        <div>
          <label className="label">Training strategy</label>
          <select className="input" value={form.training_strategy} onChange={field("training_strategy")}>
            <option value="fixed">Fixed loss weights</option>
            <option value="mgda">MGDA adaptive weights</option>
          </select>
        </div>
        <div>
          <label className="label">Validation subset</label>
          <select className="input" value={form.validation_subset} onChange={field("validation_subset")}>
            {(runtime?.available_subset_ids ?? []).map((subset) => (
              <option key={subset} value={subset}>subset{subset}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Fixed weights remain configurable; MGDA derives weights per batch. */}
      <div>
        <label className="label mb-3">{form.training_strategy === "mgda" ? "MGDA Objectives" : "Loss Weights"}</label>
        {form.training_strategy === "mgda" ? (
          <div className="space-y-3 rounded-lg border border-cyan-500/20 bg-slate-950/50 p-3">
            <p className="text-xs text-slate-300">
              MGDA combines BCE, Dice, Boundary, Focal, and differentiable HD-distance objectives. All objectives remain enabled; zero simplex weights can still occur when the measured gradients conflict.
            </p>
            <div className="grid gap-3 md:grid-cols-2">
              <div>
                <label className="label">Gradient scaling</label>
                <select className="input mt-1" value={form.mgda_gradient_normalization}
                        onChange={field("mgda_gradient_normalization")}>
                  <option value="raw">Raw task gradients</option>
                  <option value="unit">Unit-normalized task gradients</option>
                </select>
                <p className="mt-1 text-xs text-slate-500">Unit mode solves on normalized gradients and applies matching inverse-norm coefficients.</p>
              </div>
              <div>
                <label className="label">Parameters used by MGDA</label>
                <select className="input mt-1" value={form.mgda_parameter_scope}
                        onChange={field("mgda_parameter_scope")}>
                  <option value="mask_head">Mask head and predictor</option>
                  <option value="shared_mask_path">Trainable backbone and mask branch</option>
                </select>
              </div>
            </div>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-5 gap-3">
              {LW.map(({ k, label, color }) => (
                <div key={k}>
                  <label className={`label ${color}`}>{label}</label>
                  <input className="input" type="number" min={0} max={1} step={0.1}
                         value={form[k]} onChange={num(k)} />
                </div>
              ))}
            </div>
            <div className="mt-2 text-xs text-slate-400">
              Preview: {LW.filter(l => form[l.k] > 0).map(l => `${form[l.k]}·${l.label.split(" ")[1]}`).join(" + ") || "None"}
            </div>
          </>
        )}
      </div>

      {/* Training params */}
      <div className="grid grid-cols-3 gap-4">
        <div>
          <label className="label">Epochs</label>
          <input className="input" type="number" value={form.epochs} onChange={num("epochs")} />
        </div>
        <div>
          <label className="label">Batch Size</label>
          <input className="input" type="number" value={form.batch_size} onChange={num("batch_size")} />
        </div>
        <div>
          <label className="label">Learning Rate</label>
          <input className="input" type="number" step="0.001" value={form.lr} onChange={num("lr")} />
        </div>
      </div>

      <div>
        <label className="label">Training subsets (comma-separated)</label>
        <input className="input font-mono" value={form.train_subsets} onChange={field("train_subsets")}
               placeholder="1" />
        <div className="mt-1 text-xs text-slate-500">Available subsets: {(runtime?.available_subset_ids ?? []).join(", ") || "checking…"}</div>
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        <div>
          <label className="label">Independent test subset (optional)</label>
          <select className="input" value={form.test_subset} onChange={field("test_subset")}>
            <option value="">Unavailable / not selected</option>
            {(runtime?.available_subset_ids ?? []).map((subset) => (
              <option key={subset} value={subset}>subset{subset}</option>
            ))}
          </select>
          <div className="mt-1 text-xs text-slate-500">No independent test subset is configured by default.</div>
        </div>
        <div>
          <label className="label">Custom data root (optional)</label>
          <input className="input w-full font-mono" value={form.data_root} onChange={field("data_root")}
                 placeholder={runtime?.default_data_root || "data/luna16"} />
          <div className="mt-1 text-xs text-slate-500">Default backend path: {runtime?.default_data_root || "loading…"}</div>
        </div>
      </div>

      {err && (
        <div className="flex items-center gap-2 text-sm text-red-400">
          <AlertCircle size={15} /> {err}
        </div>
      )}

      <button className="btn-primary flex items-center gap-2 w-fit" onClick={submit} disabled={loading || loadingInfo}>
        {loading ? <Loader2 size={14} className="animate-spin" /> : <Plus size={14} />}
        Create Experiment
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Experiment Row with expandable learning curve
// ---------------------------------------------------------------------------
function ExperimentRow({ exp, onDelete, onStart }) {
  const [expanded, setExpanded] = useState(false);
  const [details, setDetails] = useState(exp);
  const [runs, setRuns] = useState([]);
  const [logs, setLogs] = useState("");
  const [loading, setLoading] = useState(false);
  const [detailError, setDetailError] = useState("");
  const [streamState, setStreamState] = useState("idle");
  const logRef = useRef(null);
  const displayStatus = details?.status ?? exp.status;

  useEffect(() => {
    if (!expanded) return undefined;
    let active = true;
    let inFlight = false;
    let fallbackTimer = null;
    let startupTimer = null;
    let socket = null;
    const refresh = async () => {
      if (inFlight || !active) return;
      inFlight = true;
      try {
        const snapshot = await api.getExperimentDetails(exp.id);
        if (!active) return;
        setDetails(snapshot.experiment);
        setRuns(snapshot.runs ?? []);
        setLogs(snapshot.log ?? "");
        setDetailError("");
      } catch (error) {
        if (active) setDetailError(error.message);
      } finally {
        inFlight = false;
        if (active) setLoading(false);
      }
    };
    const enableFallback = () => {
      if (!active || fallbackTimer) return;
      setStreamState("polling");
      fallbackTimer = window.setInterval(refresh, 5000);
      void refresh();
    };
    setLoading(true);
    void refresh();
    if (displayStatus === "running") {
      setStreamState("connecting");
      socket = openTrainingWS(exp.id, (message) => {
        if (!message?.ping && active) {
          setStreamState("connected");
          void refresh();
        }
      }, () => enableFallback(), () => {
        if (startupTimer) window.clearTimeout(startupTimer);
        setStreamState("connected");
      });
      startupTimer = window.setTimeout(() => {
        if (socket?.readyState !== WebSocket.OPEN) enableFallback();
      }, 2500);
    } else {
      setStreamState("idle");
    }
    return () => {
      active = false;
      if (startupTimer) window.clearTimeout(startupTimer);
      if (fallbackTimer) window.clearInterval(fallbackTimer);
      if (socket && socket.readyState < WebSocket.CLOSING) socket.close();
    };
  }, [expanded, exp.id, displayStatus]);

  useEffect(() => {
    const el = logRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    if (atBottom) el.scrollTop = el.scrollHeight;
  }, [logs]);

  const chartData = runs.map((run) => ({
    epoch: run.epoch,
    "Train Loss": run.train_total_loss,
    "Val Dice": run.val_dice,
    "HD95": run.val_hd95,
    "Boundary F1": run.val_boundary_f1,
  }));
  const config = details?.config_json ?? {};
  const artifactSummary = config.artifact_summary ?? {};
  const finalMetrics = artifactSummary.final_metrics ?? {};
  const weights = ["alpha", "beta", "gamma", "delta", "epsilon"];
  const lossNames = { alpha: "BCE", beta: "Dice", gamma: "Boundary", delta: "Focal", epsilon: "HD95" };
  const shown = (value, digits = 4) => Number.isFinite(value) ? value.toFixed(digits) : "—";
  const toggleExpand = () => setExpanded((value) => !value);

  return (
    <div className="card">
      <div
        className="flex items-center gap-3 cursor-pointer"
        onClick={toggleExpand}
      >
        {expanded ? <ChevronDown size={16} className="text-slate-400" /> : <ChevronRight size={16} className="text-slate-400" />}
        <span className="font-mono text-sm text-cyan-400 flex-1 truncate">{exp.config_id}</span>
        <span className="hidden md:block text-xs text-slate-400 flex-1">{exp.loss_description}</span>
        <StatusBadge status={displayStatus} />
        <div className="flex gap-2 ml-2" onClick={(e) => e.stopPropagation()}>
          {displayStatus === "pending" && (
            <button className="btn-primary py-1 px-3 flex items-center gap-1 text-xs"
                    onClick={() => onStart(exp.id)}>
              <Play size={12} /> Start
            </button>
          )}
          <button className="btn-secondary py-1 px-2 text-xs"
                  onClick={() => onDelete(exp.id)}
                  disabled={displayStatus === "running"}>
            <Trash2 size={12} />
          </button>
        </div>
      </div>

      {/* Loss weights pills */}
      <div className="flex flex-wrap gap-2 mt-3 pl-6">
        {weights.map((k) => (
          exp[k] > 0 && (
            <span key={k} className="text-xs bg-slate-800 px-2 py-0.5 rounded text-slate-300">
              {k[0].toUpperCase()} = {exp[k]}
            </span>
          )
        ))}
      </div>

      {/* Expandable curve */}
      {expanded && (
        <div className="mt-5 space-y-5 border-t border-slate-800 pt-5">
          {loading && <div className="text-xs text-slate-400 flex items-center gap-2"><Loader2 size={12} className="animate-spin" /> Loading run details...</div>}
          {detailError && <div className="text-sm text-red-400">{detailError}</div>}
          {displayStatus === "running" && <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-cyan-500/15 bg-cyan-500/5 px-3 py-2 text-[11px]"><span className="text-cyan-100">Live experiment monitor</span><span className={streamState === "connected" ? "text-emerald-300" : streamState === "polling" ? "text-amber-200" : "text-slate-400"}>{streamState === "connected" ? "WebSocket connected" : streamState === "polling" ? "WebSocket unavailable · polling every 5 seconds" : "Connecting to progress stream…"}</span><span className="text-slate-400">Epoch {runs.at(-1)?.epoch ?? 0}/{config.epochs ?? "—"} · loss {shown(runs.at(-1)?.train_total_loss)} · pseudo-mask Dice {shown(runs.at(-1)?.val_dice)} · HD95 {shown(runs.at(-1)?.val_hd95, 2)} mm · LR {shown(runs.at(-1)?.lr, 6)}</span></div>}

          <section>
            <h3 className="text-sm font-semibold text-white mb-3">Run configuration</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
              {[
                ["Phase", details?.phase], ["Training strategy", config.training_strategy ?? "fixed"],
                ["Validation fold", config.fold],
                ["Epochs", config.epochs], ["Batch size", config.batch_size],
                ["Learning rate", config.lr], ["Data root", config.data_root],
                ["Loading mode", config.cache_mode === "cached" ? "Cached NPZ (fast path)" : "Raw CT volumes"],
                ["Cache directory", config.cache_dir], ["Patch size", config.patch_size],
                ["Training subsets", config.train_subsets?.join(", ")],
                ["Validation subset", config.validation_subset],
                ["Test subset", config.test_subset ?? "Unavailable / not selected"],
                ["Cache patches", config.cache_summary?.total_patches],
                ["Cache spacing", config.cache_summary?.spacing_mm?.join(" × ")],
                ["Training scans", config.data_split?.train_scan_uids?.length],
                ["Validation scans", config.data_split?.validation_scan_uids?.length],
                ["Test scans", config.data_split?.test_scan_uids?.length ?? 0],
                ["Artifact folder", config.artifact_dir ?? config.source_folder],
                ["Created", details?.created_at], ["Started", details?.started_at],
                ["Finished", details?.finished_at],
              ].map(([label, value]) => (
                <div key={label} className="rounded bg-slate-900/70 p-3 min-w-0">
                  <div className="text-slate-500 mb-1">{label}</div>
                  <div className="text-slate-200 break-all">{value ? String(value) : "—"}</div>
                </div>
              ))}
            </div>
            {(artifactSummary.best_dice != null || Object.keys(finalMetrics).length > 0) && (
              <div className="mt-4">
                <h4 className="text-xs font-semibold text-slate-200 mb-2">Saved artifact summary</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                  {[
                    ["Best epoch", artifactSummary.best_epoch],
                    ["Best pseudo-mask Dice", artifactSummary.best_dice],
                    ["Final epoch", finalMetrics.epoch],
                    ["Final train loss", finalMetrics.total_loss],
                    ["Final validation loss", finalMetrics.val_total ?? finalMetrics.val_total_loss],
                    ["Final pseudo-mask Dice", finalMetrics.val_dice],
                    ["Final IoU", finalMetrics.val_iou],
                    ["Final HD95 (mm)", finalMetrics.val_hd95],
                    ["Final Boundary F1", finalMetrics.val_boundary_f1],
                  ].filter(([, value]) => value != null).map(([label, value]) => (
                    <div key={label} className="rounded bg-slate-900/70 p-3">
                      <div className="text-slate-500 mb-1">{label}</div>
                      <div className="text-slate-200">{typeof value === "number" ? shown(value, label.includes("epoch") ? 0 : 4) : String(value)}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {config.data_split && (
              <details className="mt-3 rounded-lg border border-slate-800 bg-slate-950/55 p-3">
                <summary className="cursor-pointer text-xs font-semibold text-cyan-200">
                  Resolved scan-level split (train {config.data_split.train_scan_uids?.length ?? 0}, validation {config.data_split.validation_scan_uids?.length ?? 0}, test {config.data_split.test_scan_uids?.length ?? 0})
                </summary>
                <div className="mt-3 grid md:grid-cols-3 gap-3 text-[10px] text-slate-400">
                  {[
                    ["Training UIDs", config.data_split.train_scan_uids],
                    ["Validation UIDs", config.data_split.validation_scan_uids],
                    ["Test UIDs", config.data_split.test_scan_uids],
                  ].map(([label, uids]) => (
                    <div key={label} className="max-h-48 overflow-auto rounded bg-black/30 p-2">
                      <div className="mb-1 font-semibold text-slate-300">{label}</div>
                      {(uids ?? []).length ? uids.join("\n") : "No independent test scans configured"}
                    </div>
                  ))}
                </div>
                {!config.data_split.independent_test_available && (
                  <p className="mt-2 text-[10px] text-amber-300">No independent test set is available for this experiment.</p>
                )}
              </details>
            )}
            <div className="flex flex-wrap gap-2 mt-3">
              {config.training_strategy === "mgda" ? (
                <span className="rounded bg-cyan-950/70 px-2 py-1 text-xs text-cyan-200">
                  Adaptive weights, component losses, and shared gradient norms are logged per epoch.
                </span>
              ) : weights.map((key) => (
                <span key={key} className="rounded bg-slate-800 px-2 py-1 text-xs text-slate-300">
                  {lossNames[key]}: {details?.[key] ?? "—"}
                </span>
              ))}
            </div>
          </section>

          <section>
            <h3 className="text-sm font-semibold text-white mb-3">Epoch metrics ({runs.length})</h3>
            {chartData.length > 0 && (
              <div className="mb-4">
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
                    <XAxis dataKey="epoch" tick={{ fill: "#64748b", fontSize: 10 }} />
                    <YAxis tick={{ fill: "#64748b", fontSize: 10 }} />
                    <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }}
                             labelStyle={{ color: "#cbd5e1" }} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Line type="monotone" dataKey="Train Loss" stroke="#f87171" strokeWidth={1.5} dot={false} />
                    <Line type="monotone" dataKey="Val Dice" name="Pseudo-mask Dice" stroke="#34d399" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="Boundary F1" stroke="#a78bfa" strokeWidth={1.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
            {runs.length > 0 ? (
              <div className="overflow-x-auto rounded border border-slate-800">
                <table className="min-w-[1450px] w-full text-xs">
                  <thead className="bg-slate-900 text-slate-400">
                    <tr>{["Epoch", "LR", "Train total", "Validation loss", "BCE", "Dice loss", "Boundary", "Focal", "HD loss", "Pseudo-mask Dice", "IoU", "HD95", "Boundary F1", "Sensitivity", "Specificity", "ASD", "Volume error %"].map((h) => <th key={h} className="p-2 text-right first:text-left">{h}</th>)}</tr>
                  </thead>
                  <tbody>
                    {runs.map((run) => (
                      <tr key={run.id} className="border-t border-slate-800/70 text-slate-300">
                        <td className="p-2">{run.epoch}</td>
                        {[run.lr, run.train_total_loss, run.val_total_loss, run.train_bce_loss, run.train_dice_loss,
                          run.train_boundary_loss, run.train_focal_loss, run.train_hd_loss,
                          run.val_dice, run.val_iou, run.val_hd95, run.val_boundary_f1,
                          run.val_sensitivity, run.val_specificity, run.val_asd, run.val_vol_err]
                          .map((value, i) => <td key={i} className="p-2 text-right">{shown(value, i === 0 ? 6 : 4)}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <p className="text-xs text-slate-500">No epoch metrics recorded yet.</p>}
          </section>

          <section>
            <div className="flex items-center gap-2 mb-3">
              <Terminal size={15} className="text-green-400" />
              <h3 className="text-sm font-semibold text-white">Experiment terminal output</h3>
              <span className="text-xs text-slate-500">{displayStatus === "running" ? "Live · refreshes every 5 seconds" : "Saved run log"}</span>
            </div>
            <pre ref={logRef} className="max-h-96 min-h-28 overflow-auto rounded-lg border border-slate-800 bg-black/80 p-4 text-xs leading-5 text-green-300 whitespace-pre-wrap break-words">
              {logs || "No run output captured yet. Start this experiment to see its logs here."}
            </pre>
          </section>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Experiments Page
// ---------------------------------------------------------------------------
export default function Experiments() {
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading]         = useState(true);
  const [showForm, setShowForm]       = useState(false);
  const [filterStatus, setFilter]     = useState("");
  const [loadError, setLoadError]     = useState("");
  const [syncing, setSyncing]         = useState(false);
  const [syncMessage, setSyncMessage] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.listExperiments(filterStatus ? { status: filterStatus } : {});
      setExperiments(data);
      setLoadError("");
    } catch (error) {
      setLoadError(error.message || "Could not load experiments");
    } finally {
      setLoading(false);
    }
  }, [filterStatus]);

  useEffect(() => { load(); }, [load]);

  const handleSync = async () => {
    setSyncing(true);
    setSyncMessage("");
    try {
      const result = await api.syncExperimentArtifacts();
      await load();
      setSyncMessage(
        `Scanned ${result.discovered} saved runs: ${result.created} added, ${result.updated} refreshed, ${result.skipped_running} active runs preserved.`
      );
    } catch (error) {
      setLoadError(error.message || "Could not sync saved experiment runs");
    } finally {
      setSyncing(false);
    }
  };

  const handleStart = async (id) => {
    try { await api.startExperiment(id); load(); }
    catch (e) { alert(e.message); }
  };

  const handleDelete = async (id) => {
    if (!confirm("Delete this experiment?")) return;
    try { await api.deleteExperiment(id); load(); }
    catch (e) { alert(e.message); }
  };

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex-1">
          <h2 className="text-lg font-bold text-white">Loss Function Experiments</h2>
          <p className="text-sm text-slate-400">Configure α·BCE + β·Dice + γ·Boundary weights</p>
        </div>
        <select className="input w-40" value={filterStatus} onChange={e => setFilter(e.target.value)}>
          <option value="">All Status</option>
          {["pending","running","done","failed"].map(s => <option key={s} value={s}>{s}</option>)}
        </select>
        <button className="btn-secondary flex items-center gap-2" onClick={handleSync} disabled={syncing}>
          {syncing ? <Loader2 size={15} className="animate-spin" /> : <RefreshCw size={15} />}
          {syncing ? "Syncing saved runs" : "Sync saved runs"}
        </button>
        <button className="btn-primary flex items-center gap-2" onClick={() => setShowForm((v) => !v)}>
          <Plus size={15} /> New Experiment
        </button>
      </div>

      <div className="rounded-lg border border-amber-400/20 bg-amber-400/[.06] px-4 py-3 text-xs leading-5 text-amber-100/90" role="note">
        Segmentation metrics (Dice, IoU, HD95, Boundary F1, and related scores) measure agreement with generated spherical/elliptical pseudo-masks on sampled patches. They are not expert-contour accuracy or clinical segmentation performance.
      </div>

      {/* Create form */}
      {showForm && <CreateForm onCreated={() => { load(); setShowForm(false); }} />}

      {loadError && <div className="card border-rose-500/20 text-sm text-rose-200" role="alert">{loadError}</div>}
      {syncMessage && <div className="card border-cyan-500/20 text-sm text-cyan-100" role="status">{syncMessage}</div>}

      {/* List */}
      {loading ? (
        <div className="flex justify-center py-16"><Loader2 className="animate-spin text-cyan-400" size={28} /></div>
      ) : (
        <div className="space-y-3">
          {experiments.map((e) => (
            <ExperimentRow key={e.id} exp={e} onStart={handleStart} onDelete={handleDelete} />
          ))}
          {experiments.length === 0 && (
            <div className="card text-center py-12 text-slate-500">
              No experiments found. Create one above ↑
            </div>
          )}
        </div>
      )}
    </div>
  );
}
