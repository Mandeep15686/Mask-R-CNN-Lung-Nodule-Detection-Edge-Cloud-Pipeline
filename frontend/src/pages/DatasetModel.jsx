import React, { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { AlertTriangle, Database, Loader2, ScanLine, Server } from "lucide-react";
import { api } from "../services/api.js";

const value = (item) => item == null || item === "" ? "—" : String(item);

function Info({ label, value: content }) {
  return <div className="rounded-lg border border-slate-700/40 bg-slate-950/35 p-3"><div className="text-[10px] uppercase tracking-wide text-slate-500">{label}</div><div className="mt-1 break-all text-xs text-slate-200">{value(content)}</div></div>;
}

export default function DatasetModel() {
  const [dataset, setDataset] = useState(null);
  const [summary, setSummary] = useState(null);
  const [model, setModel] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    Promise.allSettled([api.getDatasetStatus(), api.getDashboardSummary(), api.getInferenceStatus()])
      .then((results) => {
        if (results[0].status === "fulfilled") setDataset(results[0].value);
        if (results[1].status === "fulfilled") setSummary(results[1].value);
        if (results[2].status === "fulfilled") setModel(results[2].value);
        const failures = results.filter((result) => result.status === "rejected").map((result) => result.reason?.message);
        setError(failures.filter(Boolean).join(" · "));
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="flex min-h-64 items-center justify-center gap-3 text-sm text-slate-400"><Loader2 size={18} className="animate-spin text-cyan-300" /> Loading dataset and model metadata…</div>;
  const stats = dataset?.cache_summary;
  const models = (summary?.recent_experiments || []).filter((experiment) => experiment.checkpoint_available);

  return <div className="space-y-6">
    <header><div className="text-[10px] uppercase tracking-[.18em] text-cyan-200/70">Dataset and model registry</div><h2 className="mt-1 text-xl font-semibold text-slate-100">Data provenance & checkpoints</h2><p className="mt-1 text-sm text-slate-400">Runtime metadata from the backend cache validator and checkpoint loader.</p></header>
    {error && <div className="card flex gap-2 text-sm text-amber-200"><AlertTriangle size={16} />Some registry sources are unavailable: {error}</div>}

    <section className="card">
      <div className="flex flex-wrap items-center justify-between gap-3"><div><div className="flex items-center gap-2"><Database size={16} className="text-cyan-200" /><h3 className="text-sm font-semibold text-slate-100">LUNA16 V3 patch cache</h3></div><p className="mt-1 text-xs text-slate-500">Statistics are read from the current cache manifest and referenced patch contents.</p></div><span className={`badge ${dataset?.cache_valid ? "badge-green" : dataset?.cache_exists ? "badge-orange" : "badge-red"}`}>{dataset?.cache_valid ? "Validated" : dataset?.cache_exists ? "Invalid" : "Unavailable"}</span></div>
      <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4"><Info label="Data root" value={dataset?.default_data_root} /><Info label="Cache directory" value={dataset?.default_cache_dir} /><Info label="Manifest" value={dataset?.cache_valid ? "Valid and structurally checked" : dataset?.cache_error || "Unavailable"} /><Info label="Patch dimensions / channels" value={stats?.patch_dimensions ? `${stats.patch_dimensions.slice(1).join("×")} / ${stats.channel_count ?? stats.patch_dimensions[0]}` : null} /><Info label="Total patches" value={stats?.total_patches} /><Info label="Positive / negative" value={stats ? `${stats.positive_patches} / ${stats.negative_patches}` : null} /><Info label="Physical spacing" value={stats?.spacing_mm ? `${stats.spacing_mm.join(" × ")} mm` : "Not recorded"} /><Info label="Available subsets" value={stats?.available_subsets?.map((id) => `subset${id}`).join(", ")} /></div>
      {stats?.subsets && <div className="mt-4 overflow-x-auto"><table className="w-full min-w-[500px] text-xs"><thead><tr className="border-b border-slate-700/50 text-left text-[10px] uppercase text-slate-500"><th className="py-2">Subset</th><th className="py-2">Patches</th><th className="py-2">Positive</th><th className="py-2">Negative</th><th className="py-2">Scans</th></tr></thead><tbody>{Object.entries(stats.subsets).map(([subset, row]) => <tr key={subset} className="border-b border-slate-800/50"><td className="py-2 font-mono text-cyan-200">subset{subset}</td><td className="py-2">{row.patch_count}</td><td className="py-2">{row.positive_patches}</td><td className="py-2">{row.negative_patches}</td><td className="py-2">{row.scan_uid_count}</td></tr>)}</tbody></table></div>}
      <div className="mt-4 grid gap-3 md:grid-cols-3"><Info label="Default training split" value={dataset?.default_split?.train_subsets?.map((id) => `subset${id}`).join(", ")} /><Info label="Default validation split" value={dataset?.default_split?.validation_subset == null ? null : `subset${dataset.default_split.validation_subset}`} /><Info label="Independent test data" value={dataset?.default_split?.test_subset == null ? "Unavailable" : `subset${dataset.default_split.test_subset}`} /></div>
      {!dataset?.default_split?.test_subset && <p className="mt-3 text-xs text-amber-200">This local installation has no independent test subset. Validation results must not be described as independent test results.</p>}
      <p className="mt-2 text-xs leading-5 text-amber-200">Positive segmentation masks are generated from nodule center and diameter as spherical/elliptical approximations. Scores against these pseudo-masks do not establish expert-contour accuracy.</p>
    </section>

    <section className="card">
      <div className="flex flex-wrap items-start justify-between gap-3"><div><div className="flex items-center gap-2"><Server size={16} className="text-purple-200" /><h3 className="text-sm font-semibold text-slate-100">Active inference model</h3></div><p className="mt-1 text-xs text-slate-500">The backend’s default model remains unchanged until an experiment checkpoint is explicitly selected for inference.</p></div><span className={`badge ${model?.model_loaded ? "badge-green" : "badge-red"}`}>{model?.model_loaded ? "Loaded" : "Unavailable"}</span></div>
      <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4"><Info label="Architecture" value="Mask R-CNN ResNet-50-FPN" /><Info label="Checkpoint" value={model?.checkpoint_path} /><Info label="Associated experiment" value={model?.config_id || "Not identified"} /><Info label="Device" value={model?.device} /></div>
      {!model?.model_loaded && <p className="mt-3 text-xs text-amber-200">{model?.model_loading_error || "No trained checkpoint is loaded. Inference is unavailable until a checkpoint is configured."}</p>}
    </section>

    <section className="card">
      <div className="flex items-center gap-2"><ScanLine size={16} className="text-cyan-200" /><h3 className="text-sm font-semibold text-slate-100">Experiment checkpoint registry</h3></div>
      <p className="mt-1 text-xs text-slate-500">Only checkpoints that currently exist on the backend are listed. Selecting one in Inference does not replace the active default model.</p>
      {models.length ? <div className="mt-4 overflow-x-auto"><table className="w-full min-w-[850px] text-xs"><thead><tr className="border-b border-slate-700/50 text-left text-[10px] uppercase text-slate-500"><th className="py-2">Model / experiment</th><th className="py-2">State</th><th className="py-2">Best pseudo-mask Dice</th><th className="py-2">Best epoch</th><th className="py-2">Cache</th><th className="py-2">Action</th></tr></thead><tbody>{models.map((experiment) => <tr key={experiment.id} className="border-b border-slate-800/50"><td className="py-3"><div className="font-mono text-cyan-200">{experiment.config_id}</div><div className="mt-1 max-w-[420px] truncate text-slate-500">{experiment.checkpoint_path}</div></td><td className="py-3">{experiment.status}</td><td className="py-3">{experiment.best_dice == null ? "—" : Number(experiment.best_dice).toFixed(4)}</td><td className="py-3">{experiment.best_epoch ?? "—"}</td><td className="py-3">{experiment.cache_mode || "—"}</td><td className="py-3"><button className="btn btn-secondary" onClick={() => navigate(`/inference?experiment_id=${experiment.id}`)}>Select for inference</button></td></tr>)}</tbody></table></div> : <div className="research-empty mt-4">No experiment best_model.pth checkpoints were found under the configured experiment folders.</div>}
      <Link to="/experiments" className="mt-4 inline-flex text-xs text-cyan-200 hover:text-white">Open experiments →</Link>
    </section>
    <div className="text-[10px] text-slate-600">This page reports file and loader state. Checkpoint presence alone is not treated as proof that a model loaded successfully.</div>
  </div>;
}
