import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { Activity, BrainCircuit, Loader2, RefreshCw } from "lucide-react";
import { api } from "../services/api.js";
import { StatusBadge } from "./Dashboard.jsx";

const TASKS = ["bce", "dice", "boundary", "focal", "hd"];
const n = (value, digits = 4) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
const strategyOf = (item) => item.config_json?.training_strategy ?? item.config_json?.strategy;

function Metric({ label, value, hint }) {
  return <div className="mini-metric"><span>{label}</span><strong>{value}</strong>{hint && <small className="mt-1 block text-slate-500">{hint}</small>}</div>;
}

export default function MGDASection() {
  const [items, setItems] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const all = await api.listExperiments();
      const mgda = all.filter((item) => strategyOf(item) === "mgda" || item.config_json?.use_mgda === true || /\bMGDA\b/i.test(item.loss_description || ""));
      setItems(mgda);
      setSelectedId((current) => mgda.some((item) => String(item.id) === current) ? current : String(mgda[0]?.id ?? ""));
      setError("");
    } catch (e) {
      setError(e.message || "Could not load MGDA runs.");
    } finally { setLoading(false); }
  }, []);

  useEffect(() => { void load(); }, [load]);

  useEffect(() => {
    let active = true;
    if (!selectedId) { setDetail(null); return () => { active = false; }; }
    setDetailLoading(true);
    api.getExperimentDetails(selectedId)
      .then((value) => { if (active) setDetail(value); })
      .catch((e) => { if (active) setError(e.message || "Could not load MGDA run details."); })
      .finally(() => { if (active) setDetailLoading(false); });
    return () => { active = false; };
  }, [selectedId]);

  const history = useMemo(() => detail?.history?.length ? detail.history : detail?.runs ?? [], [detail]);
  const best = detail?.artifact_summary?.best_metrics ?? {};
  const experiment = detail?.experiment;
  const config = experiment?.config_json ?? {};
  const splitInfo = config.data_split ?? config.data_split_summary ?? {};
  const selected = items.find((item) => String(item.id) === selectedId);
  const completed = items.filter((item) => item.status === "done").length;
  const bestDice = Math.max(0, ...items.map((item) => Number(item.config_json?.artifact_summary?.best_dice ?? item.config_json?.artifact_summary?.best_metrics?.val_dice ?? 0)));

  return <div className="space-y-5">
    <header className="flex flex-wrap items-start justify-between gap-3">
      <div><div className="flex items-center gap-2 text-cyan-200"><BrainCircuit size={18} /><span className="text-[10px] uppercase tracking-[.2em]">Phase 4 · Multi-objective training</span></div>
        <h2 className="mt-1 text-xl font-bold text-white">MGDA experiments</h2>
        <p className="mt-1 max-w-3xl text-sm text-slate-400">Review adaptive simplex weights, applied loss coefficients, task gradient magnitudes, conflict diagnostics, validation metrics, and the saved run log for each MGDA experiment.</p></div>
      <button className="btn btn-secondary flex items-center gap-2" onClick={load} disabled={loading}><RefreshCw size={14} className={loading ? "animate-spin" : ""} />Refresh</button>
    </header>

    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
      <Metric label="MGDA runs" value={items.length} hint="Persisted MGDA strategy records" />
      <Metric label="Completed" value={completed} hint="Status marked done" />
      <Metric label="Best recorded Dice" value={n(bestDice)} hint="Validation pseudo-mask agreement" />
      <Metric label="Active objectives" value="BCE · Dice · Boundary · Focal · HD" hint="HD is a differentiable surrogate" />
    </div>

    <section className="card space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3"><div><h3 className="text-sm font-semibold text-white">Saved MGDA runs</h3><p className="text-xs text-slate-500">Select a run to inspect its configuration, epoch diagnostics, and terminal output.</p></div>
        {items.length > 0 && <select className="input min-w-64" aria-label="Select MGDA run" value={selectedId} onChange={(e) => setSelectedId(e.target.value)}>{items.map((item) => <option key={item.id} value={item.id}>{item.config_id} · {item.status}</option>)}</select>}
      </div>
      {error && <div className="rounded-lg border border-rose-400/20 bg-rose-500/5 p-3 text-xs text-rose-200" role="alert">{error}</div>}
      {loading ? <div className="research-empty flex items-center justify-center gap-2"><Loader2 size={16} className="animate-spin" />Loading MGDA run records…</div>
        : items.length ? <div className="overflow-x-auto"><table className="w-full min-w-[760px] text-xs"><thead><tr>{["Run", "Status", "Epochs", "Parameter scope", "Normalization", "Best Dice", "Best HD95", "Boundary F1"].map((label) => <th key={label} className="px-3 py-2 text-left font-medium text-slate-500">{label}</th>)}</tr></thead><tbody>{items.map((item) => { const cfg = item.config_json ?? {}; const s = cfg.artifact_summary ?? {}; const bm = s.best_metrics ?? {}; return <tr key={item.id} onClick={() => setSelectedId(String(item.id))} className={`cursor-pointer border-t border-slate-800/70 ${selectedId === String(item.id) ? "bg-cyan-950/30" : "hover:bg-slate-900/40"}`}><td className="px-3 py-3 font-mono text-cyan-200">{item.config_id}</td><td className="px-3 py-3"><StatusBadge status={item.status} /></td><td className="px-3 py-3">{cfg.epochs ?? "—"}</td><td className="px-3 py-3">{cfg.mgda_parameter_scope ?? cfg.mgda_param_scope ?? "—"}</td><td className="px-3 py-3">{cfg.mgda_gradient_normalization ?? "—"}</td><td className="px-3 py-3 text-emerald-200">{n(s.best_dice ?? bm.val_dice)}</td><td className="px-3 py-3 text-amber-200">{n(bm.val_hd95, 2)}</td><td className="px-3 py-3 text-purple-200">{n(bm.val_boundary_f1)}</td></tr>; })}</tbody></table></div>
        : <div className="research-empty">No MGDA runs are currently indexed. Use <strong>Experiments → Sync saved runs</strong> to import saved run folders, or launch an MGDA experiment from the Experiments page.</div>}
    </section>

    {selected && <>
      <section className="card">
        <div className="flex flex-wrap items-start justify-between gap-2"><div><div className="text-[10px] uppercase tracking-[.18em] text-cyan-300">Run configuration and provenance</div><h3 className="mt-1 font-mono text-base text-white">{selected.config_id}</h3></div><StatusBadge status={selected.status} /></div>
        {detailLoading ? <div className="research-empty mt-4">Loading run details…</div> : detail && <>
          <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {[["Strategy", config.training_strategy ?? "mgda"], ["Objectives", config.mgda_tasks?.join(", ") || TASKS.join(", ")], ["Gradient normalization", config.mgda_gradient_normalization], ["Shared parameter scope", config.mgda_parameter_scope ?? config.mgda_param_scope], ["Shared parameter count", config.mgda_shared_parameter_count], ["Shared parameter values", config.mgda_shared_parameter_numel], ["Learning rate", config.lr], ["Epochs / batch", `${config.epochs ?? "—"} / ${config.batch_size ?? "—"}`], ["Cache mode", config.cache_mode], ["Training subsets", splitInfo.train_subsets?.join(", ")], ["Validation subset", splitInfo.validation_subset], ["Independent test", splitInfo.independent_test_available ? `subset${splitInfo.test_subset}` : "Unavailable"]].map(([label, value]) => <div key={label} className="rounded-lg border border-slate-800 bg-slate-950/50 p-3"><div className="text-[10px] text-slate-500">{label}</div><div className="mt-1 break-all text-xs text-slate-200">{value == null || value === "" ? "—" : String(value)}</div></div>)}
          </div>
          <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-4"><Metric label="Best pseudo-mask Dice" value={n(config.artifact_summary?.best_dice ?? best.val_dice)} hint={`Epoch ${config.artifact_summary?.best_epoch ?? detail.artifact_summary?.best_epoch ?? "—"}`} /><Metric label="Best HD95" value={n(best.val_hd95, 3)} hint="mm · validation" /><Metric label="Best Boundary F1" value={n(best.val_boundary_f1)} hint="validation" /><Metric label="Epoch history rows" value={history.length} hint="Persisted in the run artifacts" /></div>
          <p className="mt-3 text-[11px] leading-5 text-amber-200">Dice, IoU, HD95, and Boundary F1 describe agreement with generated pseudo-masks on sampled patches. No independent test set is recorded for this two-subset setup.</p>
        </>}
      </section>

      {!detailLoading && history.length > 0 && <>
        <section className="card"><div className="mb-3"><h3 className="text-sm font-semibold text-white">Validation trajectory</h3><p className="text-xs text-slate-500">Epoch-level validation metrics and training loss.</p></div><ResponsiveContainer width="100%" height={260}><LineChart data={history}><CartesianGrid stroke="#26394b" strokeDasharray="3 4" /><XAxis dataKey="epoch" tick={{ fill: "#8ea8ba", fontSize: 10 }} /><YAxis yAxisId="score" domain={[0, 1]} tick={{ fill: "#8ea8ba", fontSize: 10 }} /><YAxis yAxisId="distance" orientation="right" tick={{ fill: "#8ea8ba", fontSize: 10 }} /><Tooltip /><Legend /><Line yAxisId="score" dataKey="val_dice" name="Pseudo-mask Dice" stroke="#34d399" dot={false} /><Line yAxisId="score" dataKey="val_boundary_f1" name="Boundary F1" stroke="#38bdf8" dot={false} /><Line yAxisId="distance" dataKey="val_hd95" name="HD95 (mm)" stroke="#a78bfa" dot={false} /><Line yAxisId="distance" dataKey="total_loss" name="Train total loss" stroke="#fb923c" dot={false} /></LineChart></ResponsiveContainer></section>

        <section className="card"><div className="mb-3"><h3 className="text-sm font-semibold text-white">MGDA epoch diagnostics</h3><p className="text-xs text-slate-500">Simplex weights, applied coefficients, task losses and gradients are shown separately; conflict values are cosine similarities.</p></div><div className="overflow-x-auto"><table className="w-full min-w-[1900px] text-[10px]"><thead><tr>{["Epoch", ...TASKS.map((t) => `λ ${t}`), ...TASKS.map((t) => `Coeff ${t}`), ...TASKS.map((t) => `Loss ${t}`), ...TASKS.map((t) => `‖g ${t}‖`), "QP grad‖·‖", "Optimizer grad‖·‖", "LR × grad‖·‖", "Dice", "HD95", "Boundary F1", "Sampled negative patch FP rate"].map((label) => <th key={label} className="whitespace-nowrap px-2 py-2 text-right text-slate-500 first:text-left">{label}</th>)}</tr></thead><tbody>{history.map((row) => <tr key={row.epoch} className="border-t border-slate-800/70 text-slate-300"><td className="px-2 py-2">{row.epoch}</td>{TASKS.map((t) => <td key={`w-${t}`} className="px-2 py-2 text-right text-cyan-200">{n(row[`mgda_weight_${t}`], 5)}</td>)}{TASKS.map((t) => <td key={`c-${t}`} className="px-2 py-2 text-right">{n(row[`mgda_coefficient_${t}`], 5)}</td>)}{TASKS.map((t) => <td key={`l-${t}`} className="px-2 py-2 text-right">{n(row[`mgda_loss_${t}`], 5)}</td>)}{TASKS.map((t) => <td key={`g-${t}`} className="px-2 py-2 text-right">{n(row[`mgda_grad_norm_${t}`], 5)}</td>)}{[row.mgda_qp_gradient_norm, row.mgda_optimizer_grad_norm_post_clip, row.mgda_lr_times_postclip_grad_norm, row.val_dice, row.val_hd95, row.val_boundary_f1, row.val_sampled_negative_patch_false_positive_rate].map((value, index) => <td key={index} className="px-2 py-2 text-right">{n(value, index === 4 ? 3 : 5)}</td>)}</tr>)}</tbody></table></div>
          <details className="mt-4 rounded-lg border border-slate-800 bg-slate-950/50 p-3"><summary className="cursor-pointer text-xs font-semibold text-cyan-200">Pairwise gradient cosine and Gram diagnostics</summary><div className="mt-3 overflow-x-auto"><table className="w-full min-w-[900px] text-[10px]"><thead><tr><th className="px-2 py-2 text-left text-slate-500">Epoch</th>{TASKS.flatMap((a) => TASKS.filter((b) => TASKS.indexOf(b) > TASKS.indexOf(a)).map((b) => <th key={`${a}-${b}`} className="px-2 py-2 text-right text-slate-500">{a}/{b} cosine</th>))}</tr></thead><tbody>{history.map((row) => <tr key={row.epoch} className="border-t border-slate-800/70"><td className="px-2 py-2">{row.epoch}</td>{TASKS.flatMap((a) => TASKS.filter((b) => TASKS.indexOf(b) > TASKS.indexOf(a)).map((b) => <td key={`${a}-${b}`} className="px-2 py-2 text-right">{n(row[`mgda_cosine_${a}_${b}`], 4)}</td>))}</tr>)}</tbody></table></div></details>
        </section>
      </>}

      <section className="card"><div className="mb-2 flex items-center gap-2 text-sm font-semibold text-white"><Activity size={15} className="text-emerald-300" />Training terminal output</div><pre className="max-h-96 overflow-auto rounded-lg border border-slate-800 bg-black/70 p-3 text-[10px] leading-5 text-green-200 whitespace-pre-wrap break-words">{detail?.log || "No terminal log saved for this run."}</pre></section>
    </>}
  </div>;
}
