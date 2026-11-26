import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { FileDown, Loader2, RefreshCw, Target } from "lucide-react";
import { api } from "../services/api.js";

const WEIGHTS = ["alpha", "beta", "gamma", "delta", "epsilon"];
const LOSS_LABEL = { alpha: "BCE", beta: "Dice", gamma: "Boundary", delta: "Focal", epsilon: "HD95 surrogate" };
const pretty = (value, digits = 4) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";

function Tile({ title, value, detail, accent = "text-cyan-100" }) {
  return <div className="mini-metric"><span>{title}</span><strong className={accent}>{value}</strong>{detail && <small className="mt-1 block text-slate-500">{detail}</small>}</div>;
}

function weightTable(weights = {}) {
  return WEIGHTS.map((key) => ({ key, loss: LOSS_LABEL[key], value: weights[key] }));
}

export default function Phase3Section() {
  const [studies, setStudies] = useState([]);
  const [studyName, setStudyName] = useState("");
  const [trialNumber, setTrialNumber] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const result = await api.getPhase3Studies();
      const nextStudies = result.studies ?? [];
      setStudies(nextStudies);
      setStudyName((current) => nextStudies.some((study) => study.study_name === current) ? current : nextStudies[0]?.study_name ?? "");
      setError("");
    } catch (e) { setError(e.message || "Could not load Phase 3 study artifacts."); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  const study = studies.find((item) => item.study_name === studyName);
  const trials = study?.trials ?? [];
  const selectedTrial = trials.find((trial) => trial.trial_number === trialNumber)
    ?? trials.find((trial) => trial.trial_number === study?.best?.best_trial)
    ?? trials[0];
  const best = study?.best ?? {};
  const complete = trials.filter((trial) => trial.state === "COMPLETE").length;
  const weights = best.best_norm_weights ?? selectedTrial?.normalized_weights ?? {};
  const totalWeights = WEIGHTS.reduce((total, key) => total + (Number(weights[key]) || 0), 0);
  const chartRows = useMemo(() => trials.map((trial) => ({
    trial: trial.trial_number,
    score: trial.score,
    dice: trial.metrics?.val_dice ?? trial.metrics?.dice,
    hd95: trial.metrics?.val_hd95 ?? trial.metrics?.hd95,
    boundary: trial.metrics?.val_boundary_f1 ?? trial.metrics?.boundary_f1,
  })), [trials]);

  const exportCsv = () => {
    if (!trials.length) return;
    const cols = ["trial_number", "state", "score", "val_dice", "val_hd95", "val_boundary_f1", "best_epoch", ...WEIGHTS.map((key) => `${key}_normalized`), "elapsed_s", "checkpoint_available"];
    const rows = trials.map((trial) => [trial.trial_number, trial.state, trial.score,
      trial.metrics?.val_dice, trial.metrics?.val_hd95, trial.metrics?.val_boundary_f1,
      trial.best_epoch, ...WEIGHTS.map((key) => trial.normalized_weights?.[key]), trial.elapsed_s,
      trial.checkpoint_available]);
    const csv = [cols, ...rows].map((row) => row.map((value) => `"${String(value ?? "").replaceAll('"', '""')}"`).join(",")).join("\r\n");
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    link.download = `${studyName || "phase3"}-trials.csv`;
    link.click();
    URL.revokeObjectURL(link.href);
  };

  return <div className="space-y-5">
    <header className="flex flex-wrap items-start justify-between gap-3">
      <div><div className="flex items-center gap-2 text-purple-200"><Target size={18} /><span className="text-[10px] uppercase tracking-[.2em]">Phase 3 · Optuna TPE</span></div><h2 className="mt-1 text-xl font-bold text-white">Bayesian loss-weight search</h2><p className="mt-1 max-w-3xl text-sm text-slate-400">Inspect every persisted study and trial: normalized objective weights, validation results, trial configuration, epoch history, checkpoint status, and run output.</p></div>
      <div className="flex flex-wrap items-center gap-2">{studies.length > 0 && <select className="input min-w-56" aria-label="Select Phase 3 study" value={studyName} onChange={(e) => { setStudyName(e.target.value); setTrialNumber(null); }}>{studies.map((item) => <option key={item.study_name} value={item.study_name}>{item.study_name}</option>)}</select>}<button className="btn btn-secondary flex items-center gap-2" onClick={refresh} disabled={loading}><RefreshCw size={14} className={loading ? "animate-spin" : ""} />Refresh</button><button className="btn btn-secondary flex items-center gap-2" onClick={exportCsv} disabled={!trials.length}><FileDown size={14} />Export trials CSV</button></div>
    </header>

    {error && <div className="card border-rose-500/20 text-sm text-rose-200" role="alert">{error}</div>}
    {loading ? <div className="card research-empty flex items-center justify-center gap-2"><Loader2 className="animate-spin" size={16} />Loading persisted Phase 3 studies…</div>
      : !study ? <div className="card"><h3 className="text-sm font-semibold text-white">No Phase 3 study artifacts found</h3><p className="mt-2 text-xs leading-5 text-slate-400">The backend checks <code>EXPERIMENTS_DIR/phase3/studies</code> (default: the repository’s <code>experiments/phase3/studies</code>). Start or copy a Phase 3 study onto the backend host, then refresh this page.</p></div>
      : <>
        <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
          <Tile title="Completed trials" value={`${study.completed_trials} / ${best.total_trials ?? trials.length}`} detail={`${trials.length} trial records · ${best.pruned_trials ?? 0} pruned`} />
          <Tile title="Best trial / score" value={`#${best.best_trial ?? "—"} · ${pretty(best.best_score, 6)}`} detail="Objective maximized" accent="text-emerald-200" />
          <Tile title="Best validation Dice" value={pretty(best.best_dice)} detail={`Best epoch ${best.best_epoch ?? "—"}`} accent="text-emerald-200" />
          <Tile title="Best validation HD95" value={`${pretty(best.best_hd95, 3)} mm`} detail="Lower is better" accent="text-amber-200" />
          <Tile title="Best Boundary F1" value={pretty(best.best_boundary_f1)} detail="Validation pseudo-mask agreement" accent="text-purple-200" />
        </section>

        <section className="card">
          <div className="flex flex-wrap items-start justify-between gap-3"><div><div className="text-[10px] uppercase tracking-[.18em] text-purple-300">Study and search configuration</div><h3 className="mt-1 text-sm font-semibold text-white">{study.study_name}</h3></div><span className="rounded-full border border-emerald-400/20 bg-emerald-400/5 px-3 py-1 text-xs text-emerald-200">{complete === (best.total_trials ?? trials.length) ? "Target complete" : "Study has incomplete trials"}</span></div>
          <div className="mt-4 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">{[["Optimizer", "Optuna TPE"], ["Sampler seed", best.seed ?? selectedTrial?.training_config?.seed ?? "Not recorded"], ["Trials / epochs", `${best.total_trials ?? trials.length} / ${selectedTrial?.training_config?.epochs ?? 10}`], ["Parallel jobs", "1 · sequential GPU trials"], ["Objective", best.objective_formula || "0.5×Dice + 0.3×(1−HD95/100) + 0.2×Boundary F1"], ["Learning rate", best.best_lr ?? selectedTrial?.training_config?.lr], ["BCE positive weight", best.best_bce_pos_weight ?? selectedTrial?.training_config?.bce_pos_weight], ["Best checkpoint", selectedTrial?.checkpoint_available ? `Available for selected trial #${selectedTrial.trial_number}` : "Not present for selected trial"], ["Training subsets", selectedTrial?.training_config?.train_subsets?.map((value) => `subset${value}`).join(", ")], ["Validation subset", selectedTrial?.training_config?.validation_subset == null ? "Not recorded" : `subset${selectedTrial.training_config.validation_subset}`], ["Independent test", selectedTrial?.training_config?.independent_test_available ? `subset${selectedTrial.training_config.test_subset}` : "Unavailable"], ["Selected trial data", `${selectedTrial?.training_config?.train_scan_count ?? "—"} training scans · ${selectedTrial?.training_config?.validation_scan_count ?? "—"} validation scans`]].map(([label, value]) => <div key={label} className="rounded-lg border border-slate-800 bg-slate-950/50 p-3"><div className="text-[10px] text-slate-500">{label}</div><div className="mt-1 break-all text-xs text-slate-200">{value == null || value === "" ? "—" : String(value)}</div></div>)}</div>
          <div className="mt-4"><h4 className="mb-2 text-xs font-semibold text-slate-200">Best trial normalized loss weights · sum {pretty(totalWeights, 6)}</h4><div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-5">{weightTable(weights).map(({ key, loss, value }) => <div key={key} className="rounded-lg border border-slate-800 bg-slate-950/60 p-3"><div className="text-[10px] text-slate-500">{loss}</div><div className="mt-1 text-lg font-semibold text-cyan-100">{pretty(value, 6)}</div><div className="mt-2 h-1.5 rounded-full bg-slate-800"><div className="h-full rounded-full bg-cyan-400" style={{ width: `${Math.max(0, Math.min(100, Number(value) * 100))}%` }} /></div></div>)}</div></div>
          <div className="mt-4 rounded-lg border border-amber-300/20 bg-amber-300/[.05] p-3 text-[11px] leading-5 text-amber-100">These segmentation metrics are validation agreement with generated pseudo-masks on sampled patches. This run uses the available train/validation subsets and has no independent test subset; it should not be presented as independent test performance.</div>
        </section>

        <section className="card"><div className="mb-3"><h3 className="text-sm font-semibold text-white">Trial search trajectory</h3><p className="text-xs text-slate-500">Each point is one persisted trial. Select a row below to inspect its epoch-level history.</p></div><ResponsiveContainer width="100%" height={280}><LineChart data={chartRows}><CartesianGrid stroke="#26394b" strokeDasharray="3 4" /><XAxis dataKey="trial" tick={{ fill: "#8ea8ba", fontSize: 10 }} label={{ value: "Trial", position: "insideBottom", offset: -3, fill: "#8ea8ba", fontSize: 11 }} /><YAxis yAxisId="score" domain={[0, 1]} tick={{ fill: "#8ea8ba", fontSize: 10 }} /><YAxis yAxisId="hd95" orientation="right" tick={{ fill: "#8ea8ba", fontSize: 10 }} /><Tooltip /><Legend /><Line yAxisId="score" connectNulls dataKey="score" name="Objective score" stroke="#38bdf8" strokeWidth={2} dot={{ r: 2 }} /><Line yAxisId="score" connectNulls dataKey="dice" name="Best validation Dice" stroke="#34d399" dot={false} /><Line yAxisId="score" connectNulls dataKey="boundary" name="Boundary F1" stroke="#a78bfa" dot={false} /><Line yAxisId="hd95" connectNulls dataKey="hd95" name="HD95 (mm)" stroke="#fb923c" dot={false} /></LineChart></ResponsiveContainer></section>

        <section className="card"><div className="mb-3"><h3 className="text-sm font-semibold text-white">All trials ({trials.length})</h3><p className="text-xs text-slate-500">Metrics and weights are loaded from the study CSV and each trial’s saved artifacts where available.</p></div><div className="overflow-x-auto"><table className="w-full min-w-[1350px] text-[10px]"><thead><tr>{["Trial", "State", "Score", "Best epoch", "Dice", "HD95 mm", "Boundary F1", ...WEIGHTS.map((key) => LOSS_LABEL[key]), "Time min", "Checkpoint"].map((label) => <th key={label} className="whitespace-nowrap px-2 py-2 text-right text-slate-500 first:text-left">{label}</th>)}</tr></thead><tbody>{trials.map((trial) => <tr key={trial.trial_number} onClick={() => setTrialNumber(trial.trial_number)} className={`cursor-pointer border-t border-slate-800/70 ${selectedTrial?.trial_number === trial.trial_number ? "bg-purple-950/30" : "hover:bg-slate-900/40"}`}><td className="px-2 py-2 text-left font-mono text-cyan-200">{String(trial.trial_number).padStart(2, "0")}</td><td className={`px-2 py-2 text-right ${trial.state === "COMPLETE" ? "text-emerald-200" : trial.state === "FAIL" ? "text-rose-300" : "text-amber-200"}`}>{trial.state}</td><td className="px-2 py-2 text-right">{pretty(trial.score, 6)}</td><td className="px-2 py-2 text-right">{trial.best_epoch ?? "—"}</td><td className="px-2 py-2 text-right">{pretty(trial.metrics?.val_dice ?? trial.metrics?.dice)}</td><td className="px-2 py-2 text-right">{pretty(trial.metrics?.val_hd95 ?? trial.metrics?.hd95, 3)}</td><td className="px-2 py-2 text-right">{pretty(trial.metrics?.val_boundary_f1 ?? trial.metrics?.boundary_f1)}</td>{WEIGHTS.map((key) => <td key={key} className="px-2 py-2 text-right">{pretty(trial.normalized_weights?.[key], 5)}</td>)}<td className="px-2 py-2 text-right">{pretty(Number(trial.elapsed_s) / 60, 1)}</td><td className="px-2 py-2 text-right">{trial.checkpoint_available ? "Saved" : "—"}</td></tr>)}</tbody></table></div></section>

        {selectedTrial && <section className="card space-y-4"><div className="flex flex-wrap items-start justify-between gap-2"><div><div className="text-[10px] uppercase tracking-[.18em] text-cyan-300">Trial detail</div><h3 className="mt-1 font-mono text-sm text-white">Trial #{selectedTrial.trial_number} · {selectedTrial.state}</h3></div><span className="text-xs text-slate-400">{selectedTrial.checkpoint_available ? "Best checkpoint saved" : "Checkpoint not found"}</span></div>
          <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">{[["Objective score", selectedTrial.score], ["Best epoch", selectedTrial.best_epoch], ["Training epochs", selectedTrial.training_config?.epochs], ["Elapsed time", `${pretty(Number(selectedTrial.elapsed_s) / 60, 1)} min`], ["Learning rate", selectedTrial.training_config?.lr], ["Batch size", selectedTrial.training_config?.batch_size], ["Warmup epochs", selectedTrial.training_config?.warmup_epochs], ["Cache mode", selectedTrial.training_config?.cache_mode]].map(([label, value]) => <div key={label} className="rounded-lg border border-slate-800 bg-slate-950/50 p-3"><div className="text-[10px] text-slate-500">{label}</div><div className="mt-1 text-xs text-slate-200">{value == null ? "—" : String(value)}</div></div>)}</div>
          <div className="grid gap-4 xl:grid-cols-2"><div><h4 className="mb-2 text-xs font-semibold text-slate-200">Raw sampled weights</h4><div className="grid grid-cols-2 gap-2">{weightTable(selectedTrial.raw_weights).map(({ key, loss, value }) => <div key={key} className="rounded bg-slate-950/60 p-2 text-xs"><span className="text-slate-500">{loss}</span><span className="float-right text-slate-200">{pretty(value, 6)}</span></div>)}</div></div><div><h4 className="mb-2 text-xs font-semibold text-slate-200">Normalized weights</h4><div className="grid grid-cols-2 gap-2">{weightTable(selectedTrial.normalized_weights).map(({ key, loss, value }) => <div key={key} className="rounded bg-slate-950/60 p-2 text-xs"><span className="text-slate-500">{loss}</span><span className="float-right text-cyan-100">{pretty(value, 6)}</span></div>)}</div><div className="mt-2 text-right text-[10px] text-slate-500">sum = {pretty(WEIGHTS.reduce((sum, key) => sum + (Number(selectedTrial.normalized_weights?.[key]) || 0), 0), 6)}</div></div></div>
          <div><h4 className="mb-2 text-xs font-semibold text-slate-200">Epoch history · selected trial</h4>{selectedTrial.history?.length ? <div className="overflow-x-auto"><table className="w-full min-w-[1150px] text-[10px]"><thead><tr>{["Epoch", "LR", "Train loss", "Validation loss", "Val BCE", "Val Boundary", "Val Focal", "Val HD surrogate", "Pseudo-mask Dice", "IoU", "HD95 mm", "Boundary F1", "Sensitivity", "Specificity"].map((label) => <th key={label} className="whitespace-nowrap px-2 py-2 text-right text-slate-500 first:text-left">{label}</th>)}</tr></thead><tbody>{selectedTrial.history.map((row) => <tr key={row.epoch} className="border-t border-slate-800/70"><td className="px-2 py-2">{row.epoch}</td>{[row.lr,row.total_loss,row.val_total,row.val_bce,row.val_boundary,row.val_focal,row.val_hd,row.val_dice,row.val_iou,row.val_hd95,row.val_boundary_f1,row.val_sensitivity,row.val_specificity].map((value,index)=><td key={index} className="px-2 py-2 text-right">{pretty(value,index===0?6:4)}</td>)}</tr>)}</tbody></table></div> : <p className="research-empty">No epoch history file was saved for this trial.</p>}</div>
          <details className="rounded-lg border border-slate-800 bg-slate-950/50 p-3"><summary className="cursor-pointer text-xs font-semibold text-slate-200">Trial terminal output</summary><pre className="mt-3 max-h-80 overflow-auto whitespace-pre-wrap break-words text-[10px] leading-5 text-green-200">{selectedTrial.log || "No terminal log saved."}</pre></details>
        </section>}
      </>}
  </div>;
}
