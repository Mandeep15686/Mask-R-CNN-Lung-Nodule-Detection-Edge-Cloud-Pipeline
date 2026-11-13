import React, { useEffect, useState, useCallback } from "react";
import { api, openTrainingWS } from "../services/api.js";
import { Plus, Play, Trash2, ChevronDown, ChevronRight, Loader2, AlertCircle } from "lucide-react";
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
  const [curve, setCurve]       = useState(null);
  const [wsData, setWsData]     = useState([]);
  const [loading, setLoading]   = useState(false);

  const toggleExpand = async () => {
    if (!expanded && !curve) {
      setLoading(true);
      try {
        const data = await api.getLearningCurve(exp.id);
        const rows = data.epochs.map((ep, i) => ({
          epoch: ep,
          "Train Loss": data.train_loss[i],
          "Val Dice":   data.val_dice[i],
          "HD95":       data.val_hd95[i],
          "Boundary F1": data.val_boundary_f1[i],
        }));
        setCurve(rows);
      } catch {}
      setLoading(false);
    }
    setExpanded((e) => !e);
  };

  // Live WS for running experiments
  useEffect(() => {
    if (exp.status !== "running") return;
    const ws = openTrainingWS(exp.id, (msg) => {
      if (msg.epoch) {
        setWsData((d) => [...d, {
          epoch: msg.epoch,
          "Val Dice": msg.metrics?.val_dice,
          "Train Loss": msg.metrics?.total_loss,
          "HD95": msg.metrics?.val_hd95,
        }]);
      }
    }, () => {});
    return () => ws.close();
  }, [exp.id, exp.status]);

  const chartData = wsData.length > 0 ? wsData : (curve ?? []);

  return (
    <div className="card">
      <div
        className="flex items-center gap-3 cursor-pointer"
        onClick={toggleExpand}
      >
        {expanded ? <ChevronDown size={16} className="text-slate-400" /> : <ChevronRight size={16} className="text-slate-400" />}
        <span className="font-mono text-sm text-cyan-400 flex-1 truncate">{exp.config_id}</span>
        <span className="hidden md:block text-xs text-slate-400 flex-1">{exp.loss_description}</span>
        <StatusBadge status={exp.status} />
        <div className="flex gap-2 ml-2" onClick={(e) => e.stopPropagation()}>
          {exp.status === "pending" && (
            <button className="btn-primary py-1 px-3 flex items-center gap-1 text-xs"
                    onClick={() => onStart(exp.id)}>
              <Play size={12} /> Start
            </button>
          )}
          <button className="btn-secondary py-1 px-2 text-xs"
                  onClick={() => onDelete(exp.id)}
                  disabled={exp.status === "running"}>
            <Trash2 size={12} />
          </button>
        </div>
      </div>

      {/* Loss weights pills */}
      <div className="flex flex-wrap gap-2 mt-3 pl-6">
        {["alpha","beta","gamma","delta","epsilon"].map((k) => (
          exp[k] > 0 && (
            <span key={k} className="text-xs bg-slate-800 px-2 py-0.5 rounded text-slate-300">
              {k[0].toUpperCase()} = {exp[k]}
            </span>
          )
        ))}
      </div>

      {/* Expandable curve */}
      {expanded && (
        <div className="mt-4 pl-6">
          {loading && <div className="text-xs text-slate-400 flex items-center gap-2"><Loader2 size={12} className="animate-spin" /> Loading...</div>}
          {!loading && chartData.length > 0 && (
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
                <XAxis dataKey="epoch" tick={{ fill: "#64748b", fontSize: 10 }} />
                <YAxis tick={{ fill: "#64748b", fontSize: 10 }} />
                <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }}
                         labelStyle={{ color: "#cbd5e1" }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="Train Loss" stroke="#f87171" strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="Val Dice"   stroke="#34d399" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="Boundary F1" stroke="#a78bfa" strokeWidth={1.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          )}
          {!loading && chartData.length === 0 && (
            <div className="text-xs text-slate-500">No epoch data yet — start the experiment first.</div>
          )}
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
