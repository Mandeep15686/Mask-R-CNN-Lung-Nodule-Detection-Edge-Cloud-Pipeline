import React, { useEffect, useMemo, useState } from "react";
import { Download, Loader2, Star } from "lucide-react";
import {
  Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Scatter,
  ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../services/api.js";

const number = (value, digits = 4) => value == null || !Number.isFinite(Number(value)) ? "—" : Number(value).toFixed(digits);
const tooltipStyle = { contentStyle: { background: "#0b1a2a", border: "1px solid rgba(110,200,224,.25)", borderRadius: 12 }, labelStyle: { color: "#d7edf5" } };
const COLUMNS = [
  ["rank", "#", 0], ["config_id", "Experiment", null], ["phase", "Phase", 0], ["status", "Status", null],
  ["alpha", "α", 2], ["beta", "β", 2], ["gamma", "γ", 2], ["delta", "δ", 2], ["epsilon", "ε", 2],
  ["best_dice", "Dice ↑", 4], ["best_iou", "IoU ↑", 4], ["best_hd95", "HD95 mm ↓", 2],
  ["best_boundary_f1", "Boundary F1 ↑", 4], ["best_sensitivity", "Sensitivity ↑", 4], ["best_specificity", "Specificity ↑", 4],
  ["best_asd", "ASD mm ↓", 2], ["best_volume_error", "Volume error %", 2], ["best_epoch", "Epoch", 0], ["training_duration_s", "Duration s", 1],
];

function exportCsv(rows) {
  const fields = COLUMNS.map(([key]) => key);
  const escape = (value) => `"${String(value ?? "").replaceAll('"', '""')}"`;
  const csv = [fields.map(escape).join(","), ...rows.map((row) => fields.map((field) => escape(row[field])).join(","))].join("\r\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = "experiment-comparison.csv"; anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export default function Leaderboard() {
  const [rows, setRows] = useState([]);
  const [pareto, setPareto] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [phase, setPhase] = useState("");
  const [status, setStatus] = useState("");
  const [sortKey, setSortKey] = useState("rank");
  const [sortDesc, setSortDesc] = useState(false);
  const [selectedIds, setSelectedIds] = useState([]);

  useEffect(() => {
    let active = true;
    setLoading(true);
    const params = { ...(phase ? { phase: Number(phase) } : {}), ...(status ? { status } : {}) };
    Promise.all([api.getLeaderboard(params), api.getPareto(params)])
      .then(([leaderboard, frontier]) => {
        if (!active) return;
        setRows(leaderboard || []);
        setPareto(frontier || []);
        setSelectedIds((previous) => previous.filter((id) => (leaderboard || []).some((row) => row.experiment_id === id)));
        setError("");
      })
      .catch((reason) => { if (active) setError(reason.message || "Could not load experiment analytics"); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [phase, status]);

  const sorted = useMemo(() => [...rows].sort((a, b) => {
    const av = a[sortKey], bv = b[sortKey];
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    const cmp = typeof av === "number" && typeof bv === "number" ? av - bv : String(av).localeCompare(String(bv));
    return sortDesc ? -cmp : cmp;
  }), [rows, sortKey, sortDesc]);
  const selected = sorted.filter((row) => selectedIds.includes(row.experiment_id));
  const frontierIds = new Set(pareto.map((row) => row.experiment_id));
  const scatterAll = rows.filter((row) => row.best_dice != null && row.best_hd95 != null).map((row) => ({ ...row, x: row.best_hd95, y: row.best_dice }));
  const scatterFrontier = scatterAll.filter((row) => frontierIds.has(row.experiment_id));
  const lossWeightPoints = rows.flatMap((row) => row.best_dice == null ? [] : [
    ["α BCE", row.alpha, "#55dff0"], ["β Dice", row.beta, "#76edcf"],
    ["γ Boundary", row.gamma, "#b59bff"], ["δ Focal", row.delta, "#ffc078"], ["ε HD95", row.epsilon, "#ff8095"],
  ].map(([name, weight, color]) => ({ name, weight, dice: row.best_dice, config_id: row.config_id, color, alpha: row.alpha, beta: row.beta, gamma: row.gamma, delta: row.delta, epsilon: row.epsilon })));
  const toggle = (id) => setSelectedIds((previous) => previous.includes(id) ? previous.filter((entry) => entry !== id) : [...previous, id]);

  if (loading) return <div className="flex min-h-64 items-center justify-center gap-3 text-sm text-slate-400"><Loader2 className="animate-spin text-cyan-300" size={18} /> Loading validation comparisons…</div>;

  return <div className="space-y-6">
    <header className="flex flex-wrap items-end justify-between gap-3"><div><div className="text-[10px] uppercase tracking-[.18em] text-purple-200/70">DIP · loss and model comparison</div><h2 className="mt-1 text-xl font-semibold text-slate-100">Validation leaderboard</h2><p className="mt-1 text-sm text-slate-400">Per-experiment metrics come from the same best validation-Dice epoch. These are validation results, not independent test results.</p></div><div className="flex flex-wrap gap-2"><select className="input w-auto" aria-label="Filter by phase" value={phase} onChange={(event) => setPhase(event.target.value)}><option value="">All phases</option>{[1, 2, 3].map((value) => <option key={value} value={value}>Phase {value}</option>)}</select><select className="input w-auto" aria-label="Filter by status" value={status} onChange={(event) => setStatus(event.target.value)}><option value="">All statuses</option>{["pending", "running", "done", "failed"].map((value) => <option key={value} value={value}>{value}</option>)}</select><button className="btn btn-secondary flex items-center gap-2" onClick={() => exportCsv(rows)}><Download size={14} />Export CSV</button></div></header>
    {error && <div className="card text-sm text-rose-200">{error}</div>}

    <section className="card">
      <div className="flex flex-wrap items-center justify-between gap-2"><div><h3 className="text-sm font-semibold text-slate-100">Pareto space</h3><p className="mt-1 text-xs text-slate-500">X = HD95 (mm), lower is better · Y = validation Dice, higher is better.</p></div><span className="badge badge-cyan">{pareto.length} non-dominated</span></div>
      {scatterAll.length ? <ResponsiveContainer width="100%" height={320}><ScatterChart margin={{ left: 10, right: 18, top: 12, bottom: 12 }}><CartesianGrid stroke="#26394b" strokeDasharray="3 4" /><XAxis type="number" dataKey="x" name="HD95" unit=" mm" tick={{ fill: "#8299aa", fontSize: 10 }} label={{ value: "HD95 (mm) ↓", position: "insideBottom", offset: -4, fill: "#8299aa", fontSize: 11 }} /><YAxis type="number" dataKey="y" name="Validation Dice" domain={[0, 1]} tick={{ fill: "#8299aa", fontSize: 10 }} label={{ value: "Validation Dice ↑", angle: -90, position: "insideLeft", fill: "#8299aa", fontSize: 11 }} /><Tooltip {...tooltipStyle} formatter={(value, name) => [name === "HD95" ? `${number(value, 2)} mm` : number(value), name]} labelFormatter={(_, items) => items?.[0]?.payload?.config_id || "Experiment"} /><Legend /><Scatter name="Other experiments" data={scatterAll.filter((row) => !frontierIds.has(row.experiment_id))} fill="#56738a" /><Scatter name="Pareto frontier" data={scatterFrontier} fill="#55dff0" /></ScatterChart></ResponsiveContainer> : <div className="research-empty">No experiments have both validation Dice and HD95 measurements for this filter.</div>}
    </section>

    <section className="card">
      <div><h3 className="text-sm font-semibold text-slate-100">Loss-weight observations</h3><p className="mt-1 text-xs text-slate-500">Configuration weight vs best validation Dice. Descriptive only; these observations do not establish causal effects.</p></div>
      {lossWeightPoints.length ? <ResponsiveContainer width="100%" height={280}><ScatterChart margin={{ left: 10, right: 18, top: 12, bottom: 12 }}><CartesianGrid stroke="#26394b" strokeDasharray="3 4" /><XAxis type="number" dataKey="weight" name="Loss weight" domain={[0, 1]} tick={{ fill: "#8299aa", fontSize: 10 }} label={{ value: "Configured loss weight (0–1)", position: "insideBottom", offset: -4, fill: "#8299aa", fontSize: 11 }} /><YAxis type="number" dataKey="dice" name="Validation Dice" domain={[0, 1]} tick={{ fill: "#8299aa", fontSize: 10 }} label={{ value: "Best validation Dice", angle: -90, position: "insideLeft", fill: "#8299aa", fontSize: 11 }} /><Tooltip {...tooltipStyle} formatter={(value, name) => [number(value), name]} labelFormatter={(_, items) => { const point = items?.[0]?.payload; return `${point?.config_id || "Experiment"} · ${point?.name || "loss"}`; }} />{["α BCE", "β Dice", "γ Boundary", "δ Focal", "ε HD95"].map((name, index) => <Scatter key={name} name={name} data={lossWeightPoints.filter((point) => point.name === name)} fill={["#55dff0", "#76edcf", "#b59bff", "#ffc078", "#ff8095"][index]} />)}<Legend /></ScatterChart></ResponsiveContainer> : <div className="research-empty mt-3">No experiment has a validation Dice measurement to compare with its loss weights.</div>}
    </section>

    <section className="card">
      <div className="flex flex-wrap items-center justify-between gap-3"><div><h3 className="text-sm font-semibold text-slate-100">Select configurations to compare</h3><p className="mt-1 text-xs text-slate-500">Choose two or more runs. Each metric remains separate and keeps its unit.</p></div><span className="text-xs text-slate-500">{selectedIds.length} selected</span></div>
      <div className="mt-4 overflow-x-auto"><table className="w-full min-w-[750px] text-xs"><thead><tr className="border-b border-slate-700/50 text-left text-[10px] uppercase text-slate-500"><th className="py-2 pr-3">Select</th>{["Experiment", "Phase", "Status", "Loss weights", "Dice", "HD95 (mm)"].map((name) => <th key={name} className="py-2 pr-3">{name}</th>)}</tr></thead><tbody>{rows.map((row) => <tr key={row.experiment_id} className="border-b border-slate-800/50"><td className="py-2.5 pr-3"><input type="checkbox" aria-label={`Compare ${row.config_id}`} checked={selectedIds.includes(row.experiment_id)} onChange={() => toggle(row.experiment_id)} /></td><td className="py-2.5 pr-3 font-mono text-cyan-200">{row.config_id}</td><td className="py-2.5 pr-3">{row.phase}</td><td className="py-2.5 pr-3"><span className={`badge ${row.status === "done" ? "badge-green" : row.status === "failed" ? "badge-red" : row.status === "running" ? "badge-orange" : "badge-slate"}`}>{row.status}</span></td><td className="py-2.5 pr-3 text-slate-400">α{number(row.alpha, 2)} β{number(row.beta, 2)} γ{number(row.gamma, 2)} δ{number(row.delta, 2)} ε{number(row.epsilon, 2)}</td><td className="py-2.5 pr-3 text-emerald-200">{number(row.best_dice)}</td><td className="py-2.5 pr-3 text-amber-200">{number(row.best_hd95, 2)}</td></tr>)}</tbody></table></div>
      {!rows.length && <div className="research-empty mt-3">No experiments match the selected phase and status.</div>}
      {selected.length >= 2 && <div className="mt-5 grid gap-4 lg:grid-cols-2"><div><div className="mb-2 text-xs font-semibold text-slate-300">Validation Dice</div><ResponsiveContainer width="100%" height={230}><BarChart data={selected}><CartesianGrid stroke="#26394b" strokeDasharray="3 4" /><XAxis dataKey="config_id" tick={{ fill: "#8299aa", fontSize: 9 }} /><YAxis domain={[0, 1]} tick={{ fill: "#8299aa", fontSize: 10 }} /><Tooltip {...tooltipStyle} /><Bar dataKey="best_dice" name="Best-epoch Dice" fill="#76edcf" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></div><div><div className="mb-2 text-xs font-semibold text-slate-300">HD95 (mm)</div><ResponsiveContainer width="100%" height={230}><BarChart data={selected}><CartesianGrid stroke="#26394b" strokeDasharray="3 4" /><XAxis dataKey="config_id" tick={{ fill: "#8299aa", fontSize: 9 }} /><YAxis tick={{ fill: "#8299aa", fontSize: 10 }} /><Tooltip {...tooltipStyle} /><Bar dataKey="best_hd95" name="Best-epoch HD95 (mm)" fill="#b59bff" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></div></div>}
      {selected.length === 1 && <p className="mt-3 text-xs text-slate-500">Select at least one more experiment to draw a comparison.</p>}
      {selected.length >= 2 && <div className="mt-4 overflow-x-auto"><table className="w-full min-w-[900px] text-xs"><thead><tr className="border-b border-slate-700/50 text-left text-[10px] uppercase text-slate-500"><th className="py-2 pr-3">Metric</th>{selected.map((row) => <th key={row.experiment_id} className="py-2 pr-3 font-mono normal-case text-cyan-200">{row.config_id}</th>)}</tr></thead><tbody>{[["Dice", "best_dice", 4], ["IoU", "best_iou", 4], ["Sensitivity", "best_sensitivity", 4], ["Specificity", "best_specificity", 4], ["HD95 (mm)", "best_hd95", 2], ["ASD (mm)", "best_asd", 2], ["Boundary F1", "best_boundary_f1", 4], ["Volume error (%)", "best_volume_error", 2], ["Training duration (s)", "training_duration_s", 1]].map(([label, key, digits]) => <tr key={key} className="border-b border-slate-800/50"><td className="py-2.5 pr-3 text-slate-400">{label}</td>{selected.map((row) => <td key={row.experiment_id} className="py-2.5 pr-3 text-slate-200">{number(row[key], digits)}</td>)}</tr>)}</tbody></table></div>}
    </section>

    <section className="card">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-3"><div><h3 className="text-sm font-semibold text-slate-100">Full experiment results</h3><p className="mt-1 text-xs text-slate-500">Sortable columns · all validation metrics come from the same best-Dice epoch.</p></div><span className="text-[10px] text-slate-500">{rows.length} rows</span></div>
      <div className="overflow-x-auto"><table className="w-full min-w-[1600px] text-xs"><thead><tr className="border-b border-slate-700/50 text-left text-[10px] uppercase text-slate-500">{COLUMNS.map(([key, label]) => <th key={key} scope="col" aria-sort={sortKey === key ? sortDesc ? "descending" : "ascending" : "none"} className="py-2 pr-3"><button type="button" className="select-none hover:text-cyan-100 focus-visible:outline focus-visible:outline-1 focus-visible:outline-cyan-300" onClick={() => { if (sortKey === key) setSortDesc((value) => !value); else { setSortKey(key); setSortDesc(false); } }}>{label}{sortKey === key ? (sortDesc ? " ↓" : " ↑") : ""}</button></th>)}<th className="py-2">Pareto</th></tr></thead><tbody>{sorted.map((row) => <tr key={row.experiment_id} className={`border-b border-slate-800/50 ${frontierIds.has(row.experiment_id) ? "bg-cyan-400/[.035]" : ""}`}>{COLUMNS.map(([key, _label, digits]) => <td key={key} className={`py-2.5 pr-3 ${key === "config_id" ? "font-mono text-cyan-200" : key === "best_dice" ? "text-emerald-200" : key === "best_hd95" ? "text-amber-200" : "text-slate-300"}`}>{digits == null ? row[key] ?? "—" : number(row[key], digits)}</td>)}<td className="py-2.5">{frontierIds.has(row.experiment_id) && <Star size={13} className="text-cyan-200" />}</td></tr>)}</tbody></table></div>
      {!rows.length && <div className="research-empty mt-3">No validation records are available for these filters.</div>}
    </section>
    <footer className="text-[10px] text-slate-600">No causal claim is inferred from these comparisons. Independent test data is not available in the current local subset0/subset1 installation.</footer>
  </div>;
}
