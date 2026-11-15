import React, { useEffect, useState, useCallback, useRef } from "react";
import { api } from "../services/api.js";
import { Plus, Play, Trash2, ChevronDown, ChevronRight, Loader2, AlertCircle, Terminal } from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
} from "recharts";
import { StatusBadge } from "./Dashboard.jsx";

// ---------------------------------------------------------------------------
// Create Experiment Form
// ---------------------------------------------------------------------------
function CreateForm({ onCreated }) {
  const [form, setForm] = useState({
    config_id: "", phase: 1,
    alpha: 0.3, beta: 0.5, gamma: 0.5, delta: 0.0, epsilon: 0.0,
    epochs: 30, batch_size: 4, fold: 0, lr: 0.005,
    data_root: "./data/luna16",
  });
  const [loading, setLoading] = useState(false);
  const [err, setErr]         = useState(null);

  const field = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));
  const num   = (k) => (e) => setForm((f) => ({ ...f, [k]: parseFloat(e.target.value) || 0 }));

  const submit = async () => {
    if (!form.config_id.trim()) return setErr("Config ID is required");
    setLoading(true); setErr(null);
    try {
      await api.createExperiment({
        config_id: form.config_id,
        phase: Number(form.phase),
        loss_weights: { alpha: form.alpha, beta: form.beta, gamma: form.gamma,
                        delta: form.delta, epsilon: form.epsilon },
        epochs: Number(form.epochs),
        batch_size: Number(form.batch_size),
        fold: Number(form.fold),
        lr: Number(form.lr),
        data_root: form.data_root,
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

  return (
    <div className="card space-y-5">
      <h2 className="text-sm font-bold text-white">New Experiment</h2>

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
          <label className="label">Fold (0–9)</label>
          <input className="input" type="number" min={0} max={9}
                 value={form.fold} onChange={num("fold")} />
        </div>
      </div>

      {/* Loss weights */}
      <div>
        <label className="label mb-3">Loss Weights</label>
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
        <label className="label">Data Root</label>
        <input className="input font-mono" value={form.data_root} onChange={field("data_root")} />
      </div>

      {err && (
        <div className="flex items-center gap-2 text-sm text-red-400">
          <AlertCircle size={15} /> {err}
        </div>
      )}

      <button className="btn-primary flex items-center gap-2 w-fit" onClick={submit} disabled={loading}>
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
  const logRef = useRef(null);
  const displayStatus = details?.status ?? exp.status;

  useEffect(() => {
    if (!expanded) return undefined;
    let active = true;
    let inFlight = false;
    const refresh = async () => {
      if (inFlight) return;
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
    setLoading(true);
    refresh();
    const timer = displayStatus === "running" ? setInterval(refresh, 5000) : null;
    return () => {
      active = false;
      if (timer) clearInterval(timer);
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

          <section>
            <h3 className="text-sm font-semibold text-white mb-3">Run configuration</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
              {[
                ["Phase", details?.phase], ["Validation fold", config.fold],
                ["Epochs", config.epochs], ["Batch size", config.batch_size],
                ["Learning rate", config.lr], ["Data root", config.data_root],
                ["Artifact folder", config.source_folder],
                ["Created", details?.created_at], ["Started", details?.started_at],
                ["Finished", details?.finished_at],
              ].map(([label, value]) => (
                <div key={label} className="rounded bg-slate-900/70 p-3 min-w-0">
                  <div className="text-slate-500 mb-1">{label}</div>
                  <div className="text-slate-200 break-all">{value ? String(value) : "—"}</div>
                </div>
              ))}
            </div>
            <div className="flex flex-wrap gap-2 mt-3">
              {weights.map((key) => (
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
                    <Line type="monotone" dataKey="Val Dice" stroke="#34d399" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="Boundary F1" stroke="#a78bfa" strokeWidth={1.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
            {runs.length > 0 ? (
              <div className="overflow-x-auto rounded border border-slate-800">
                <table className="min-w-[1450px] w-full text-xs">
                  <thead className="bg-slate-900 text-slate-400">
                    <tr>{["Epoch", "LR", "Train total", "BCE", "Dice loss", "Boundary", "Focal", "HD loss", "Val Dice", "IoU", "HD95", "Boundary F1", "Sensitivity", "Specificity", "ASD", "Volume error %"].map((h) => <th key={h} className="p-2 text-right first:text-left">{h}</th>)}</tr>
                  </thead>
                  <tbody>
                    {runs.map((run) => (
                      <tr key={run.id} className="border-t border-slate-800/70 text-slate-300">
                        <td className="p-2">{run.epoch}</td>
                        {[run.lr, run.train_total_loss, run.train_bce_loss, run.train_dice_loss,
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

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.listExperiments(filterStatus ? { status: filterStatus } : {});
      setExperiments(data);
    } catch {}
    setLoading(false);
  }, [filterStatus]);

  useEffect(() => { load(); }, [load]);

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
        <button className="btn-primary flex items-center gap-2" onClick={() => setShowForm((v) => !v)}>
          <Plus size={15} /> New Experiment
        </button>
      </div>

      {/* Create form */}
      {showForm && <CreateForm onCreated={() => { load(); setShowForm(false); }} />}

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
