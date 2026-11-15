import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  Activity, AlertTriangle, CheckCircle2, Clock3, Cpu, Database, FileDown, FlaskConical,
  Gauge, HardDrive, ImageDown, Layers3, Loader2, Network, Printer, RefreshCw, ScanLine, Search,
  Server, XCircle,
} from "lucide-react";
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Scatter,
  ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../services/api.js";

const COLORS = { cyan: "#55dff0", aqua: "#76edcf", purple: "#b59bff", amber: "#ffc078", red: "#ff8095", blue: "#7dafff" };
const statusIcon = { healthy: CheckCircle2, degraded: AlertTriangle, unavailable: XCircle, not_configured: Clock3, unknown: AlertTriangle };
const statusClass = { healthy: "text-emerald-300", degraded: "text-amber-300", unavailable: "text-rose-300", not_configured: "text-slate-400", unknown: "text-slate-400" };
const statusLabel = { healthy: "Healthy", degraded: "Degraded", unavailable: "Unavailable", not_configured: "Not configured", unknown: "Unknown" };
const pretty = (value, digits = 3) => value == null || !Number.isFinite(Number(value)) ? "—" : Number(value).toFixed(digits);
const chartTooltip = { contentStyle: { background: "#0b1a2a", border: "1px solid rgba(110,200,224,.25)", borderRadius: 12 }, labelStyle: { color: "#d7edf5" } };

function SectionTitle({ eyebrow, title, detail, action }) {
  return <div className="flex flex-wrap items-end justify-between gap-3 mb-4"><div><div className="text-[10px] uppercase tracking-[.18em] text-cyan-200/70">{eyebrow}</div><h2 className="mt-1 text-base font-semibold text-slate-100">{title}</h2>{detail && <p className="mt-1 text-xs text-slate-400">{detail}</p>}</div>{action}</div>;
}

function MetricCard({ icon: Icon, label, value, sub, color = "cyan" }) {
  return <article className="card research-metric"><div className="flex items-start justify-between gap-3"><div><p className="text-xs text-slate-400">{label}</p><div className="mt-3 text-2xl font-semibold tracking-tight tabular-nums" style={{ color: COLORS[color] || COLORS.cyan }}>{value ?? "—"}</div><p className="mt-1 text-[11px] text-slate-500">{sub || "Recorded value"}</p></div><span className="research-icon" style={{ color: COLORS[color] || COLORS.cyan }}><Icon size={17} /></span></div></article>;
}

export function StatusBadge({ status }) {
  const classes = { pending: "badge-slate", running: "badge-orange", done: "badge-green", failed: "badge-red" };
  return <span className={`badge ${classes[status] || "badge-slate"}`}>{status || "unknown"}</span>;
}

function HealthTile({ name, item }) {
  const Icon = statusIcon[item?.state] || AlertTriangle;
  const route = name === "edge_nodes" ? "/edge" : name === "inference" || name === "checkpoint" ? "/inference" : name === "dataset_cache" ? "/experiments" : null;
  const inner = <div className="flex min-h-[84px] gap-3 p-3.5"><Icon size={17} className={`mt-0.5 shrink-0 ${statusClass[item?.state] || "text-slate-400"}`} /><div className="min-w-0"><div className="flex flex-wrap items-center gap-x-2 gap-y-1"><span className="text-xs font-semibold text-slate-200">{name.replaceAll("_", " ")}</span><span className={`text-[10px] ${statusClass[item?.state] || "text-slate-400"}`}>{statusLabel[item?.state] || "Unknown"}</span></div><p className="mt-1 break-words text-[11px] leading-relaxed text-slate-500">{item?.detail || "No status detail returned"}</p></div></div>;
  return route ? <Link to={route} className="health-tile focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-300">{inner}</Link> : <div className="health-tile">{inner}</div>;
}

function ChartPanel({ title, unit, children, className = "" }) {
  const exportPng = (event) => {
    const section = event.currentTarget.closest("section");
    const svg = section?.querySelector("svg.recharts-surface") || section?.querySelectorAll("svg")?.[1];
    if (!svg) return;
    const xml = new XMLSerializer().serializeToString(svg);
    const url = URL.createObjectURL(new Blob([xml], { type: "image/svg+xml;charset=utf-8" }));
    const image = new Image();
    image.onload = () => {
      const width = svg.viewBox?.baseVal?.width || svg.clientWidth || 800;
      const height = svg.viewBox?.baseVal?.height || svg.clientHeight || 400;
      const canvas = document.createElement("canvas");
      canvas.width = width * 2; canvas.height = height * 2;
      const context = canvas.getContext("2d");
      context.fillStyle = "#081624"; context.fillRect(0, 0, canvas.width, canvas.height);
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => {
        if (blob) {
          const pngUrl = URL.createObjectURL(blob);
          const anchor = document.createElement("a"); anchor.href = pngUrl;
          anchor.download = `${title.toLowerCase().replace(/[^a-z0-9]+/g, "-")}.png`;
          anchor.click(); URL.revokeObjectURL(pngUrl);
        }
        URL.revokeObjectURL(url);
      }, "image/png");
    };
    image.onerror = () => URL.revokeObjectURL(url);
    image.src = url;
  };
  return <section className={`card min-w-0 ${className}`}><div className="mb-3 flex flex-wrap items-baseline justify-between gap-3"><h3 className="text-sm font-semibold text-slate-100">{title}</h3><div className="flex items-center gap-2"><span className="text-[10px] text-slate-500">{unit}</span><button className="btn btn-secondary px-2 py-1" aria-label={`Export ${title} as PNG`} title="Export chart as PNG" onClick={exportPng}><ImageDown size={13} /></button></div></div>{children}</section>;
}

function downloadFile(filename, content, mime) {
  const url = URL.createObjectURL(new Blob([content], { type: mime }));
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = filename; anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export default function Dashboard() {
  const [health, setHealth] = useState(null);
  const [summary, setSummary] = useState(null);
  const [activity, setActivity] = useState([]);
  const [dataset, setDataset] = useState(null);
  const [system, setSystem] = useState(null);
  const [runs, setRuns] = useState([]);
  const [selectedExperiment, setSelectedExperiment] = useState("");
  const [loadErrors, setLoadErrors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState("created_at");
  const [sortDesc, setSortDesc] = useState(true);
  const [page, setPage] = useState(0);
  const pageSize = 8;

  const refresh = useCallback(async (initial = false) => {
    if (initial) setLoading(true);
    const requests = await Promise.allSettled([
      api.getDashboardHealth(), api.getDashboardSummary(), api.getDashboardActivity({ limit: 16 }),
      api.getDatasetStatus(), api.getSystemMetrics(),
    ]);
    const setters = [setHealth, setSummary, setActivity, setDataset, setSystem];
    const errors = [];
    requests.forEach((result, index) => {
      if (result.status === "fulfilled") setters[index](result.value);
      else errors.push(result.reason?.message || "A dashboard data source is unavailable");
    });
    setLoadErrors(errors);
    setLastUpdated(new Date());
    if (initial) setLoading(false);
  }, []);

  useEffect(() => {
    refresh(true);
    const timer = window.setInterval(() => refresh(false), 30000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const experiments = summary?.recent_experiments || [];
  useEffect(() => {
    if (!experiments.length) { setRuns([]); setSelectedExperiment(""); return; }
    const selected = experiments.find((item) => String(item.id) === selectedExperiment) || experiments[0];
    if (String(selected.id) !== selectedExperiment) setSelectedExperiment(String(selected.id));
    let active = true;
    api.getRuns(selected.id).then((data) => { if (active) setRuns(data || []); })
      .catch(() => { if (active) setRuns([]); });
    return () => { active = false; };
  }, [experiments, selectedExperiment]);

  const filteredExperiments = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return [...experiments]
      .filter((item) => !needle || `${item.config_id} ${item.status} ${item.loss_description || ""}`.toLowerCase().includes(needle))
      .sort((a, b) => {
        const av = a[sortKey]; const bv = b[sortKey];
        if (av == null && bv == null) return 0;
        if (av == null) return 1;
        if (bv == null) return -1;
        const cmp = typeof av === "number" && typeof bv === "number" ? av - bv : String(av).localeCompare(String(bv));
        return sortDesc ? -cmp : cmp;
      });
  }, [experiments, query, sortKey, sortDesc]);
  const pageCount = Math.max(1, Math.ceil(filteredExperiments.length / pageSize));
  const pageRows = filteredExperiments.slice(page * pageSize, (page + 1) * pageSize);
  useEffect(() => setPage(0), [query, sortKey, sortDesc]);

  const counts = summary?.experiment_counts || {};
  const best = summary?.best_experiment;
  const selected = experiments.find((item) => String(item.id) === selectedExperiment);
  const paretoPoints = useMemo(() => {
    const valid = experiments.filter((item) => item.status === "done" && item.best_dice != null && item.best_hd95 != null);
    return valid.map((item) => ({ ...item, pareto: !valid.some((other) => other.id !== item.id && other.best_dice >= item.best_dice && other.best_hd95 <= item.best_hd95 && (other.best_dice > item.best_dice || other.best_hd95 < item.best_hd95)) }));
  }, [experiments]);
  const healthItems = health?.statuses ? Object.entries(health.statuses) : [];
  const datasetSummary = dataset?.cache_summary;
  const exportJson = () => downloadFile(
    "lung-nodule-research-report.json",
    JSON.stringify({ generated_at: new Date().toISOString(), health, dataset, summary, activity, methodology: "Validation metrics only; independent test subset is unavailable in the current two-subset dataset." }, null, 2),
    "application/json",
  );
  const exportCsv = () => {
    const fields = ["config_id", "phase", "status", "alpha", "beta", "gamma", "delta", "epsilon", "epochs_completed", "best_epoch", "best_dice", "best_iou", "best_hd95_mm", "best_boundary_f1", "best_sensitivity", "best_specificity", "best_asd", "best_volume_error", "training_duration_s"];
    const escape = (cell) => `"${String(cell ?? "").replaceAll('"', '""')}"`;
    const rows = [fields.map(escape).join(","), ...experiments.map((item) => fields.map((field) => escape(field === "best_hd95_mm" ? item.best_hd95 : item[field])).join(","))];
    downloadFile("lung-nodule-experiment-comparison.csv", rows.join("\r\n"), "text/csv;charset=utf-8");
  };

  if (loading) return <div className="flex min-h-72 items-center justify-center gap-3 text-sm text-slate-400"><Loader2 className="animate-spin text-cyan-300" size={20} /> Loading research workspace…</div>;

  return <div className="space-y-6">
    <section className="dashboard-hero research-hero">
      <div className="hero-copy"><div className="eyebrow"><Activity size={13} /> Research command center</div><h2>DIP <span>×</span> CN Research Overview</h2><p>Segmentation experiments, model readiness and measured edge–cloud observations.</p></div>
      <div className="hero-aside"><div className="hero-orb"><Layers3 size={22} /></div><div><div className="hero-aside-label">ACTIVE PIPELINE</div><div className="hero-aside-value">{health?.architecture || "Mask R-CNN ResNet-50-FPN"}</div><div className="hero-aside-sub">{health?.device || "Device unknown"} · LUNA16 2.5D</div></div></div>
    </section>

    <div className="report-controls flex flex-wrap items-center justify-end gap-2">
      <button className="btn btn-secondary flex items-center gap-2" onClick={exportCsv}><FileDown size={13} />Export comparison CSV</button>
      <button className="btn btn-secondary flex items-center gap-2" onClick={exportJson}><FileDown size={13} />Export report JSON</button>
      <button className="btn btn-secondary flex items-center gap-2" onClick={() => window.print()}><Printer size={13} />Print / PDF</button>
    </div>

    {loadErrors.length > 0 && <div className="card flex flex-wrap items-center justify-between gap-3 border-amber-400/20 text-amber-100" role="status"><div className="flex items-center gap-2"><AlertTriangle size={16} /><span className="text-xs">Some dashboard sources did not respond. Showing available data.</span></div><button className="btn btn-secondary flex items-center gap-2" onClick={() => refresh(false)}><RefreshCw size={13} /> Retry</button></div>}

    <section className="card">
      <SectionTitle eyebrow="System readiness" title="Health and runtime" detail={`Last successful dashboard refresh: ${lastUpdated?.toLocaleString() || "—"} · overall ${statusLabel[health?.overall] || "Unknown"}`} />
      <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">{healthItems.map(([name, item]) => <HealthTile key={name} name={name} item={item} />)}</div>
    </section>

    <section>
      <SectionTitle eyebrow="Research ledger" title="Recorded activity and outcomes" detail="Counts come from persisted experiment, inference and edge-network records." />
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <MetricCard icon={FlaskConical} label="Experiments" value={counts.total ?? "—"} sub={`${counts.done ?? 0} completed`} />
        <MetricCard icon={Activity} label="Running" value={counts.running ?? "—"} sub={`${counts.pending ?? 0} pending · ${counts.failed ?? 0} failed`} color="amber" />
        <MetricCard icon={ScanLine} label="Inference requests" value={summary?.inference_requests ?? "—"} sub="Persisted predictions" color="purple" />
        <MetricCard icon={Gauge} label="Best validation Dice" value={pretty(best?.best_dice, 4)} sub={best?.config_id ? `Best epoch · ${best.best_epoch}` : "No validation result recorded"} color="aqua" />
        <MetricCard icon={HardDrive} label="Best-epoch HD95" value={best?.best_hd95 == null ? "—" : `${pretty(best.best_hd95, 2)} mm`} sub={best?.config_id || "No validation result recorded"} color="amber" />
        <MetricCard icon={Network} label="Edge / network" value={`${summary?.active_edge_nodes ?? "—"} / ${summary?.recorded_network_requests ?? "—"}`} sub="Online nodes / recorded measurements" color="blue" />
      </div>
    </section>

    <section className="grid gap-4 xl:grid-cols-[minmax(0,1.55fr)_minmax(280px,.85fr)]">
      <div className="space-y-4">
        <section className="card">
          <SectionTitle eyebrow="DIP · epoch history" title="Experiment trajectory" detail="Each chart uses its own scale and units." action={<select className="input w-auto min-w-48" aria-label="Select experiment history" value={selectedExperiment} onChange={(e) => setSelectedExperiment(e.target.value)}>{experiments.map((item) => <option key={item.id} value={item.id}>{item.config_id}</option>)}</select>} />
          {runs.length ? <div className="grid gap-3 md:grid-cols-2">
            <ChartPanel title="Training and validation loss" unit="loss"><ResponsiveContainer width="100%" height={190}><LineChart data={runs}><CartesianGrid stroke="#26394b" strokeDasharray="3 4" /><XAxis dataKey="epoch" tick={{ fill: "#8ea8ba", fontSize: 10 }} /><YAxis tick={{ fill: "#8ea8ba", fontSize: 10 }} /><Tooltip {...chartTooltip} /><Legend /><Line connectNulls dataKey="train_total_loss" name="Train loss" stroke={COLORS.amber} strokeWidth={2} dot={false} /><Line connectNulls dataKey="val_total_loss" name="Validation loss" stroke={COLORS.red} strokeWidth={2} dot={false} /></LineChart></ResponsiveContainer>{runs.every((run) => run.val_total_loss == null) && <div className="mt-2 text-[10px] text-slate-500">Validation loss is now persisted for new runs; older runs may not include it.</div>}</ChartPanel>
            <ChartPanel title="Validation overlap" unit="score (0–1)"><ResponsiveContainer width="100%" height={190}><LineChart data={runs}><CartesianGrid stroke="#26394b" strokeDasharray="3 4" /><XAxis dataKey="epoch" tick={{ fill: "#8ea8ba", fontSize: 10 }} /><YAxis domain={[0, 1]} tick={{ fill: "#8ea8ba", fontSize: 10 }} /><Tooltip {...chartTooltip} /><Legend /><Line connectNulls dataKey="val_dice" name="Dice" stroke={COLORS.aqua} strokeWidth={2} dot={false} /><Line connectNulls dataKey="val_iou" name="IoU" stroke={COLORS.blue} strokeWidth={2} dot={false} /></LineChart></ResponsiveContainer></ChartPanel>
            <ChartPanel title="Boundary distance" unit="HD95 (mm)"><ResponsiveContainer width="100%" height={190}><LineChart data={runs}><CartesianGrid stroke="#26394b" strokeDasharray="3 4" /><XAxis dataKey="epoch" tick={{ fill: "#8ea8ba", fontSize: 10 }} /><YAxis tick={{ fill: "#8ea8ba", fontSize: 10 }} /><Tooltip {...chartTooltip} /><Line connectNulls dataKey="val_hd95" name="HD95" stroke={COLORS.purple} strokeWidth={2} dot={false} /></LineChart></ResponsiveContainer></ChartPanel>
            <ChartPanel title="Boundary F1" unit="score (0–1)"><ResponsiveContainer width="100%" height={190}><LineChart data={runs}><CartesianGrid stroke="#26394b" strokeDasharray="3 4" /><XAxis dataKey="epoch" tick={{ fill: "#8ea8ba", fontSize: 10 }} /><YAxis domain={[0, 1]} tick={{ fill: "#8ea8ba", fontSize: 10 }} /><Tooltip {...chartTooltip} /><Line connectNulls dataKey="val_boundary_f1" name="Boundary F1" stroke={COLORS.cyan} strokeWidth={2} dot={false} /></LineChart></ResponsiveContainer></ChartPanel>
            <ChartPanel title="Sensitivity and specificity" unit="score (0–1)"><ResponsiveContainer width="100%" height={190}><LineChart data={runs}><CartesianGrid stroke="#26394b" strokeDasharray="3 4" /><XAxis dataKey="epoch" tick={{ fill: "#8ea8ba", fontSize: 10 }} /><YAxis domain={[0, 1]} tick={{ fill: "#8ea8ba", fontSize: 10 }} /><Tooltip {...chartTooltip} /><Legend /><Line connectNulls dataKey="val_sensitivity" name="Sensitivity" stroke={COLORS.amber} strokeWidth={2} dot={false} /><Line connectNulls dataKey="val_specificity" name="Specificity" stroke={COLORS.aqua} strokeWidth={2} dot={false} /></LineChart></ResponsiveContainer></ChartPanel>
            <ChartPanel title="Learning rate" unit="optimizer LR"><ResponsiveContainer width="100%" height={190}><LineChart data={runs}><CartesianGrid stroke="#26394b" strokeDasharray="3 4" /><XAxis dataKey="epoch" tick={{ fill: "#8ea8ba", fontSize: 10 }} /><YAxis tick={{ fill: "#8ea8ba", fontSize: 10 }} /><Tooltip {...chartTooltip} /><Line connectNulls dataKey="lr" name="Learning rate" stroke={COLORS.blue} strokeWidth={2} dot={false} /></LineChart></ResponsiveContainer></ChartPanel>
          </div> : <div className="research-empty">{selected ? `No epoch metrics have been recorded for ${selected.config_id}.` : "Create an experiment to begin recording training metrics."}</div>}
        </section>

        <section className="card">
          <SectionTitle eyebrow="DIP · configuration analysis" title="Validation Dice vs HD95" detail="Points use best validation Dice epoch values; only completed experiments with both measurements are shown. This is validation data, not independent test performance." />
          {paretoPoints.length ? <ResponsiveContainer width="100%" height={280}><ScatterChart margin={{ left: 10, right: 20, top: 10, bottom: 10 }}><CartesianGrid stroke="#26394b" strokeDasharray="3 4" /><XAxis type="number" dataKey="best_hd95" name="HD95" unit=" mm" tick={{ fill: "#8ea8ba", fontSize: 10 }} label={{ value: "HD95 (mm) · lower is better", position: "insideBottom", offset: -3, fill: "#8ea8ba", fontSize: 11 }} /><YAxis type="number" dataKey="best_dice" name="Validation Dice" domain={[0, 1]} tick={{ fill: "#8ea8ba", fontSize: 10 }} label={{ value: "Validation Dice", angle: -90, position: "insideLeft", fill: "#8ea8ba", fontSize: 11 }} /><Tooltip {...chartTooltip} formatter={(value, name) => [name === "HD95" ? `${pretty(value, 2)} mm` : pretty(value, 4), name]} labelFormatter={(_, items) => items?.[0]?.payload?.config_id || "Experiment"} /><Scatter name="Experiments" data={paretoPoints.filter((point) => !point.pareto)} fill="#557087" /><Scatter name="Pareto frontier" data={paretoPoints.filter((point) => point.pareto)} fill={COLORS.cyan} /></ScatterChart></ResponsiveContainer> : <div className="research-empty">No completed experiments have both validation Dice and HD95 values yet.</div>}
        </section>
      </div>

      <aside className="space-y-4">
        <section className="card">
          <SectionTitle eyebrow="Best checkpoint" title={best?.config_id || "No scored experiment"} detail={best ? `Epoch ${best.best_epoch} · same-epoch metrics` : "Validation results will appear here when recorded."} />
          {best && <><div className="grid grid-cols-2 gap-2"><div className="mini-metric"><span>Dice</span><strong>{pretty(best.best_dice, 4)}</strong></div><div className="mini-metric"><span>IoU</span><strong>{pretty(best.best_iou, 4)}</strong></div><div className="mini-metric"><span>HD95</span><strong>{best.best_hd95 == null ? "—" : `${pretty(best.best_hd95, 2)} mm`}</strong></div><div className="mini-metric"><span>Boundary F1</span><strong>{pretty(best.best_boundary_f1, 4)}</strong></div><div className="mini-metric"><span>Sensitivity</span><strong>{pretty(best.best_sensitivity, 4)}</strong></div><div className="mini-metric"><span>Specificity</span><strong>{pretty(best.best_specificity, 4)}</strong></div></div><div className="mt-3 rounded-lg bg-slate-950/50 p-3 text-[11px] text-slate-400"><div>Weights · α {pretty(best.alpha, 2)} · β {pretty(best.beta, 2)} · γ {pretty(best.gamma, 2)} · δ {pretty(best.delta, 2)} · ε {pretty(best.epsilon, 2)}</div><div className="mt-1 break-all">Checkpoint · {best.checkpoint_available ? best.checkpoint_path : "Best checkpoint file not found"}</div><div className="mt-1">Split · train {best.data_split?.train_subsets?.map((s) => `subset${s}`).join(", ") || "—"} / validation subset{best.data_split?.validation_subset ?? "—"} / test {best.data_split?.independent_test_available ? `subset${best.data_split.test_subset}` : "unavailable"}</div></div></>}
        </section>

        <section className="card">
          <SectionTitle eyebrow="Dataset provenance" title="LUNA16 patch cache" detail="Runtime status from the backend validator." />
          <div className="flex items-center justify-between gap-2"><span className="text-xs text-slate-400">Cache state</span><span className={`badge ${dataset?.cache_valid ? "badge-green" : dataset?.cache_exists ? "badge-orange" : "badge-red"}`}>{dataset?.cache_valid ? "Validated" : dataset?.cache_exists ? "Invalid" : "Missing"}</span></div>
          <div className="mt-2 break-all font-mono text-[10px] text-cyan-200/80">{dataset?.default_cache_dir || "Cache path unavailable"}</div>
          {datasetSummary && <div className="mt-3 grid grid-cols-2 gap-x-3 gap-y-2 text-[11px]"><span className="text-slate-500">Patches</span><span>{datasetSummary.total_patches ?? "—"}</span><span className="text-slate-500">Positive / negative</span><span>{datasetSummary.positive_patches ?? "—"} / {datasetSummary.negative_patches ?? "—"}</span><span className="text-slate-500">Shape</span><span>{datasetSummary.patch_dimensions?.join("×") || "—"}</span><span className="text-slate-500">Spacing</span><span>{datasetSummary.spacing_mm?.join("×") || "—"} mm</span><span className="text-slate-500">Subsets</span><span>{datasetSummary.available_subsets?.map((s) => `subset${s}`).join(", ") || "—"}</span><span className="text-slate-500">Test split</span><span className="text-amber-200">{dataset?.default_split?.test_subset == null ? "Unavailable" : `subset${dataset.default_split.test_subset}`}</span></div>}
          {dataset?.cache_error && <p className="mt-3 break-words text-[10px] text-amber-200">{dataset.cache_error}</p>}
        </section>

        <section className="card">
          <SectionTitle eyebrow="Host telemetry" title="Hardware metrics" detail="Only measurements exposed by this host are shown." />
          <div className="space-y-2 text-xs">{[[Cpu, "CPU utilization", system?.cpu_percent == null ? "Unavailable" : `${pretty(system.cpu_percent, 1)}%`], [HardDrive, "Memory", system?.ram_used_bytes == null ? "Unavailable" : `${(system.ram_used_bytes / 1e9).toFixed(1)} / ${(system.ram_total_bytes / 1e9).toFixed(1)} GB`], [Gauge, "GPU", system?.gpu_name || "Unavailable"], [Server, "GPU memory", system?.gpu_memory_used_bytes == null ? "Unavailable" : `${(system.gpu_memory_used_bytes / 1e9).toFixed(1)} / ${(system.gpu_memory_total_bytes / 1e9).toFixed(1)} GB`]].map(([Icon, label, value]) => <div key={label} className="flex items-center justify-between gap-3"><span className="flex items-center gap-2 text-slate-500"><Icon size={13} />{label}</span><span className="text-slate-300">{value}</span></div>)}</div>
          <div className="mt-3 text-[10px] text-slate-600">GPU utilization and temperature are unavailable unless a host telemetry provider exposes them.</div>
        </section>

        <section className="card">
          <SectionTitle eyebrow="Research activity" title="Recent recorded events" />
          {activity.length ? <ol className="space-y-3">{activity.slice(0, 8).map((event, index) => <li key={`${event.timestamp}-${event.event}-${event.entity_id || index}`} className="flex gap-3"><span className={`mt-1 h-2 w-2 shrink-0 rounded-full ${event.source === "CN" || event.source === "edge" ? "bg-purple-300" : "bg-cyan-300"}`} /><div className="min-w-0"><div className="text-xs text-slate-200">{event.event}</div><div className="truncate text-[11px] text-slate-500">{event.detail}</div><div className="mt-0.5 text-[10px] text-slate-600">{new Date(event.timestamp).toLocaleString()}</div></div></li>)}</ol> : <div className="research-empty">No persisted activity is recorded yet.</div>}
        </section>
      </aside>
    </section>

    <section className="card">
      <SectionTitle eyebrow="DIP · experiment ledger" title="Recent experiments" detail="Sortable and searchable; metrics are from each run’s best validation-Dice epoch." action={<label className="relative block"><Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" /><input className="input w-56 pl-9" placeholder="Search experiments" value={query} onChange={(e) => setQuery(e.target.value)} /></label>} />
      <div className="overflow-x-auto"><table className="w-full min-w-[980px] text-xs"><thead><tr>{[["config_id", "Experiment"], ["status", "Status"], ["best_dice", "Best Dice"], ["best_hd95", "HD95 (mm)"], ["best_boundary_f1", "Boundary F1"], ["epochs_completed", "Epochs"], ["created_at", "Created"]].map(([key, label]) => <th key={key} scope="col" aria-sort={sortKey === key ? sortDesc ? "descending" : "ascending" : "none"} className="px-3 py-2 text-left text-[10px] uppercase tracking-wide text-slate-500"><button type="button" className="hover:text-cyan-100 focus-visible:outline focus-visible:outline-1 focus-visible:outline-cyan-300" onClick={() => { if (sortKey === key) setSortDesc((value) => !value); else { setSortKey(key); setSortDesc(key === "created_at"); } }}>{label}{sortKey === key ? (sortDesc ? " ↓" : " ↑") : ""}</button></th>)}</tr></thead><tbody>{pageRows.map((item) => <tr key={item.id} className="border-t border-slate-700/40"><td className="px-3 py-3"><Link to="/experiments" className="font-mono text-cyan-200 hover:text-white">{item.config_id}</Link><div className="mt-1 text-[10px] text-slate-600">α{pretty(item.alpha, 1)} · β{pretty(item.beta, 1)} · γ{pretty(item.gamma, 1)} · δ{pretty(item.delta, 1)} · ε{pretty(item.epsilon, 1)}</div></td><td className="px-3 py-3"><StatusBadge status={item.status} /></td><td className="px-3 py-3 text-emerald-200">{pretty(item.best_dice, 4)}</td><td className="px-3 py-3 text-amber-200">{item.best_hd95 == null ? "—" : pretty(item.best_hd95, 2)}</td><td className="px-3 py-3 text-purple-200">{pretty(item.best_boundary_f1, 4)}</td><td className="px-3 py-3 text-slate-300">{item.epochs_completed ?? "—"}</td><td className="px-3 py-3 text-slate-500">{item.created_at ? new Date(item.created_at).toLocaleDateString() : "—"}</td></tr>)}</tbody></table></div>
      {!pageRows.length && <div className="research-empty">No experiments match this search.</div>}
      {filteredExperiments.length > pageSize && <div className="mt-3 flex items-center justify-between text-xs text-slate-500"><span>{filteredExperiments.length} results · page {page + 1} of {pageCount}</span><div className="flex gap-2"><button className="btn btn-secondary" disabled={page === 0} onClick={() => setPage((value) => Math.max(0, value - 1))}>Previous</button><button className="btn btn-secondary" disabled={page + 1 >= pageCount} onClick={() => setPage((value) => Math.min(pageCount - 1, value + 1))}>Next</button></div></div>}
    </section>
    <footer className="pb-2 text-center text-[10px] text-slate-600">Validation results are not independent test results. Metrics are displayed only when recorded by the backend.</footer>
  </div>;
}
