import React, { useCallback, useEffect, useMemo, useState } from "react";
import { AlertTriangle, Clock3, Database, Gauge, Loader2, Network, RefreshCw, Server, Wifi, Zap } from "lucide-react";
import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../services/api.js";

const HEARTBEAT_TIMEOUT_MS = 60_000;
const tooltipStyle = { contentStyle: { background: "#0b1a2a", border: "1px solid rgba(110,200,224,.25)", borderRadius: 12 }, labelStyle: { color: "#d7edf5" } };
const mean = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
const display = (value, digits = 1) => value == null || !Number.isFinite(Number(value)) ? "—" : Number(value).toFixed(digits);

function Panel({ title, note, children }) {
  return <section className="card min-w-0"><div className="mb-3 flex flex-wrap items-baseline justify-between gap-2"><h3 className="text-sm font-semibold text-slate-100">{title}</h3>{note && <span className="text-[10px] text-slate-500">{note}</span>}</div>{children}</section>;
}

function Empty({ children = "No measurements recorded for this view yet." }) {
  return <div className="research-empty">{children}</div>;
}

function measuredValue(values, digits = 1, suffix = "") {
  const result = mean(values.filter((value) => value != null && Number.isFinite(Number(value))).map(Number));
  return result == null ? "—" : `${display(result, digits)}${suffix}`;
}

export default function EdgeNetwork() {
  const [nodes, setNodes] = useState([]);
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState(null);

  const refresh = useCallback(async (initial = false) => {
    if (initial) setLoading(true); else setRefreshing(true);
    const results = await Promise.allSettled([api.listEdgeNodes(), api.getNetworkLogs({ limit: 250 })]);
    if (results[0].status === "fulfilled") setNodes(results[0].value || []);
    if (results[1].status === "fulfilled") setLogs(results[1].value || []);
    const failures = results.filter((result) => result.status === "rejected").map((result) => result.reason?.message || "Network data unavailable");
    setError(failures.join(" · "));
    setLastUpdated(new Date());
    setLoading(false);
    setRefreshing(false);
  }, []);

  useEffect(() => {
    refresh(true);
    const timer = window.setInterval(() => refresh(false), 30000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const chronLogs = useMemo(() => [...logs].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp)), [logs]);
  const timingLogs = useMemo(
    () => logs.filter((row) => row.timing_schema_version === 2),
    [logs],
  );
  const legacyTimingCount = logs.length - timingLogs.length;
  const nodeStats = useMemo(() => {
    const grouped = new Map();
    for (const log of logs) {
      const list = grouped.get(log.node_id) || [];
      list.push(log);
      grouped.set(log.node_id, list);
    }
    const now = Date.now();
    return nodes.map((node) => {
      const nodeLogs = (grouped.get(node.node_id) || []).sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
      const lastSeen = node.last_seen ? new Date(node.last_seen).getTime() : null;
      const online = lastSeen != null && Number.isFinite(lastSeen) && now - lastSeen <= HEARTBEAT_TIMEOUT_MS;
      return {
        ...node,
        online,
        measured_latency_ms: mean(nodeLogs.filter((row) => row.timing_schema_version === 2).map((row) => Number(row.latency_ms)).filter(Number.isFinite)),
        measured_bandwidth_mbps: mean(nodeLogs.map((row) => Number(row.bandwidth_mbps)).filter(Number.isFinite)),
        last_request: nodeLogs[0] || null,
      };
    });
  }, [nodes, logs]);

  const timeline = chronLogs.map((log) => ({
    time: log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : "Unknown time",
    network_rtt_ms: log.timing_schema_version === 2 ? log.latency_ms : null,
    edge_processing_ms: log.timing_schema_version === 2 ? log.edge_processing_ms : null,
    bandwidth_mbps: log.bandwidth_mbps,
    payload_kb: log.payload_size_kb,
    cloud_ms: log.inference_time_ms,
    split_layer: log.split_layer,
    compression_method: log.compression_method,
    compression_ratio: log.compression_ratio,
  }));
  const bySplit = useMemo(() => {
    const groups = new Map();
    for (const log of timingLogs) {
      const value = groups.get(log.split_layer) || [];
      value.push(log);
      groups.set(log.split_layer, value);
    }
    return [...groups.entries()].map(([split_layer, rows]) => ({
      split_layer,
      requests: rows.length,
      measured_rtt_ms: mean(rows.map((row) => Number(row.latency_ms)).filter(Number.isFinite)),
      edge_processing_ms: mean(rows.map((row) => Number(row.edge_processing_ms)).filter(Number.isFinite)),
      cloud_inference_ms: mean(rows.map((row) => Number(row.inference_time_ms)).filter(Number.isFinite)),
      received_payload_kb: mean(rows.map((row) => Number(row.payload_size_kb)).filter(Number.isFinite)),
    }));
  }, [timingLogs]);
  const byCompression = useMemo(() => {
    const groups = new Map();
    for (const log of logs) {
      const method = log.compression_method || "unspecified";
      const value = groups.get(method) || [];
      value.push(log);
      groups.set(method, value);
    }
    return [...groups.entries()].map(([method, rows]) => ({
      method,
      requests: rows.length,
      mean_payload_kb: mean(rows.map((row) => Number(row.payload_size_kb)).filter(Number.isFinite)),
      reported_ratio: mean(rows.map((row) => Number(row.compression_ratio)).filter(Number.isFinite)),
    }));
  }, [logs]);

  const nodeOnline = nodeStats.filter((node) => node.online).length;
  const avgLatency = timingLogs.map((row) => Number(row.latency_ms)).filter(Number.isFinite);
  const avgBandwidth = timingLogs.map((row) => Number(row.bandwidth_mbps)).filter(Number.isFinite);
  const payloads = timingLogs.map((row) => Number(row.payload_size_kb)).filter(Number.isFinite);
  if (loading) return <div className="flex min-h-64 items-center justify-center gap-3 text-sm text-slate-400"><Loader2 className="animate-spin text-cyan-300" size={20} /> Loading recorded network data…</div>;

  const xAxis = <XAxis dataKey="time" minTickGap={28} tick={{ fill: "#8299aa", fontSize: 10 }} />;
  const yAxis = <YAxis tick={{ fill: "#8299aa", fontSize: 10 }} />;

  return <div className="space-y-6">
    <header className="flex flex-wrap items-end justify-between gap-3"><div><div className="text-[10px] uppercase tracking-[.18em] text-purple-200/70">CN · edge–cloud measurement lab</div><h2 className="mt-1 text-xl font-semibold text-slate-100">Network observability</h2><p className="mt-1 text-sm text-slate-400">Prototype request observations: probe RTT, edge compute, cloud inference, bandwidth, and payload are recorded separately. This is not a controlled clinical network or energy study.</p></div><div className="flex items-center gap-3"><span className="text-[10px] text-slate-500">Updated {lastUpdated?.toLocaleTimeString() || "—"}</span><button className="btn btn-secondary flex items-center gap-2" onClick={() => refresh(false)} disabled={refreshing}><RefreshCw size={13} className={refreshing ? "animate-spin" : ""} />Refresh</button></div></header>

    {error && <div className="card flex items-center gap-2 text-sm text-amber-200"><AlertTriangle size={15} /> Partial data: {error}</div>}

    {legacyTimingCount > 0 && <div className="rounded-lg border border-amber-300/20 bg-amber-300/[.04] px-4 py-3 text-xs leading-relaxed text-amber-100">{legacyTimingCount} older log row(s) have ambiguous timing semantics and are excluded from RTT and edge-processing averages. New rows use separated timing fields.</div>}

    {!logs.length && <div className="card flex items-start gap-3 border border-amber-300/20 bg-amber-300/[.04] text-sm">
      <AlertTriangle size={17} className="mt-0.5 shrink-0 text-amber-200" />
      <div><div className="font-medium text-amber-100">No network events have been recorded yet.</div>
        <p className="mt-1 text-xs leading-relaxed text-slate-400">{nodeStats.length === 0
          ? "Start the backend and edge node, then submit a scan to create the first measurement."
          : nodeOnline === 0
            ? "The registered edge node is offline. Start or restart it, then submit a scan; its probe and routing event will appear here when the backend is reachable."
            : "The edge node is online. Submit a scan to record probe bandwidth, latency, payload size, and the selected inference path."}</p>
      </div>
    </div>}

    <section className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-5">
      {[[Wifi, "Online nodes", `${nodeOnline} / ${nodeStats.length}`, "Heartbeat within 60 seconds"], [Network, "Logged edge events", logs.length, "Probe results and request routing events"], [Gauge, "Mean probe RTT", measuredValue(avgLatency, 1, " ms"), "Health endpoint round-trip time"], [Zap, "Mean bandwidth", measuredValue(avgBandwidth, 2, " Mbps"), "Recorded upload probe measurements"], [Database, "Mean received payload", measuredValue(payloads, 1, " KB"), "Measured request body size"]].map(([Icon, label, value, note]) => <article key={label} className="card"><div className="flex items-start justify-between gap-2"><div><div className="text-xs text-slate-400">{label}</div><div className="mt-3 text-xl font-semibold text-cyan-100 tabular-nums">{value}</div><div className="mt-1 text-[10px] text-slate-600">{note}</div></div><Icon size={17} className="text-cyan-200/80" /></div></article>)}
    </section>

    <section>
      <div className="mb-3 flex items-end justify-between gap-3"><div><div className="text-[10px] uppercase tracking-[.16em] text-slate-500">Heartbeat timeout · {HEARTBEAT_TIMEOUT_MS / 1000}s</div><h3 className="mt-1 text-sm font-semibold text-slate-100">Registered edge nodes</h3></div></div>
      {nodeStats.length ? <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">{nodeStats.map((node) => <article key={node.node_id} className="card"><div className="flex items-start justify-between gap-3"><div className="min-w-0"><h4 className="truncate text-sm font-semibold text-slate-100">{node.node_id}</h4><p className="mt-1 truncate text-xs text-slate-500">{node.hostname || "Unknown host"} · {node.ip_address || "IP unavailable"}</p></div><span className={`badge ${node.online ? "badge-green" : "badge-red"}`}>{node.online ? "Online" : "Offline"}</span></div><div className="mt-4 grid grid-cols-2 gap-3 text-xs"><div><div className="text-slate-500">Split layer</div><div className="mt-1 font-mono text-cyan-200">{node.split_layer || "—"}</div></div><div><div className="text-slate-500">Observed compression</div><div className="mt-1 text-slate-300">{node.last_request?.compression_method || "Not reported"}</div></div><div><div className="text-slate-500">Mean probe RTT</div><div className="mt-1 text-amber-200">{node.measured_latency_ms == null ? "—" : `${display(node.measured_latency_ms)} ms`}</div></div><div><div className="text-slate-500">Measured bandwidth mean</div><div className="mt-1 text-emerald-200">{node.measured_bandwidth_mbps == null ? "—" : `${display(node.measured_bandwidth_mbps, 2)} Mbps`}</div></div></div><div className="mt-3 border-t border-slate-700/40 pt-3 text-[10px] text-slate-500">Last heartbeat: {node.last_seen ? new Date(node.last_seen).toLocaleString() : "Never"}<br />Last successful logged request: {node.last_request?.timestamp ? new Date(node.last_request.timestamp).toLocaleString() : "None recorded"}</div></article>)}</div> : <Empty>No edge nodes are registered in the backend database.</Empty>}
    </section>

    <section className="grid gap-4 lg:grid-cols-2">
      <Panel title="Network probe RTT" note="Health endpoint round-trip · milliseconds">{timeline.length ? <ResponsiveContainer width="100%" height={235}><LineChart data={timeline}>{xAxis}{yAxis}<CartesianGrid stroke="#26394b" strokeDasharray="3 4" /><Tooltip {...tooltipStyle} /><Line connectNulls type="monotone" dataKey="network_rtt_ms" name="Probe RTT" stroke="#ffc078" strokeWidth={2} dot={false} unit=" ms" /></LineChart></ResponsiveContainer> : <Empty />}</Panel>
      <Panel title="Edge processing time" note="Preprocessing + feature extraction/compression · milliseconds">{timeline.length ? <ResponsiveContainer width="100%" height={235}><LineChart data={timeline}>{xAxis}{yAxis}<CartesianGrid stroke="#26394b" strokeDasharray="3 4" /><Tooltip {...tooltipStyle} /><Line connectNulls type="monotone" dataKey="edge_processing_ms" name="Edge processing" stroke="#55dff0" strokeWidth={2} dot={false} unit=" ms" /></LineChart></ResponsiveContainer> : <Empty />}</Panel>
      <Panel title="Bandwidth over recorded requests" note="Mbps">{timeline.length ? <ResponsiveContainer width="100%" height={235}><LineChart data={timeline}>{xAxis}{yAxis}<CartesianGrid stroke="#26394b" strokeDasharray="3 4" /><Tooltip {...tooltipStyle} /><Line connectNulls type="monotone" dataKey="bandwidth_mbps" name="Bandwidth" stroke="#76edcf" strokeWidth={2} dot={false} unit=" Mbps" /></LineChart></ResponsiveContainer> : <Empty />}</Panel>
      <Panel title="Received payload size" note="Measured HTTP body · KB">{timeline.length ? <ResponsiveContainer width="100%" height={235}><LineChart data={timeline}>{xAxis}{yAxis}<CartesianGrid stroke="#26394b" strokeDasharray="3 4" /><Tooltip {...tooltipStyle} /><Line connectNulls type="monotone" dataKey="payload_kb" name="Received payload" stroke="#b59bff" strokeWidth={2} dot={false} unit=" KB" /></LineChart></ResponsiveContainer> : <Empty />}</Panel>
      <Panel title="Cloud processing time" note="Backend inference · milliseconds">{timeline.length ? <ResponsiveContainer width="100%" height={235}><LineChart data={timeline}>{xAxis}{yAxis}<CartesianGrid stroke="#26394b" strokeDasharray="3 4" /><Tooltip {...tooltipStyle} /><Line connectNulls type="monotone" dataKey="cloud_ms" name="Cloud inference" stroke="#7dafff" strokeWidth={2} dot={false} unit=" ms" /></LineChart></ResponsiveContainer> : <Empty />}</Panel>
    </section>

    <section className="grid gap-4 lg:grid-cols-2">
      <Panel title="Split-layer timing components" note="Each series is a separate millisecond measurement; no summed end-to-end estimate">{bySplit.length ? <ResponsiveContainer width="100%" height={260}><BarChart data={bySplit}><CartesianGrid stroke="#26394b" strokeDasharray="3 4" /><XAxis dataKey="split_layer" tick={{ fill: "#8299aa", fontSize: 10 }} /><YAxis tick={{ fill: "#8299aa", fontSize: 10 }} /><Tooltip {...tooltipStyle} /><Legend /><Bar dataKey="measured_rtt_ms" name="Network probe RTT (ms)" fill="#55dff0" radius={[4, 4, 0, 0]} /><Bar dataKey="edge_processing_ms" name="Edge processing (ms)" fill="#76edcf" radius={[4, 4, 0, 0]} /><Bar dataKey="cloud_inference_ms" name="Cloud inference (ms)" fill="#b59bff" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer> : <Empty />}</Panel>
      <Panel title="Payload and compression reports" note="Payload size is measured; ratio derived from actual wire bytes">{byCompression.length ? <div className="overflow-x-auto"><table className="w-full min-w-[400px] text-xs"><thead><tr className="border-b border-slate-700/60 text-left text-[10px] uppercase text-slate-500"><th className="py-2 pr-3">Method</th><th className="py-2 pr-3">Requests</th><th className="py-2 pr-3">Mean received payload</th><th className="py-2">Wire-verified ratio</th></tr></thead><tbody>{byCompression.map((row) => <tr key={row.method} className="border-b border-slate-800/60"><td className="py-3 pr-3 font-mono text-purple-200">{row.method}</td><td className="py-3 pr-3">{row.requests}</td><td className="py-3 pr-3">{display(row.mean_payload_kb)} KB</td><td className="py-3">{row.reported_ratio == null ? "—" : `${display(row.reported_ratio, 2)}×`}</td></tr>)}</tbody></table></div> : <Empty />}</Panel>
    </section>

    <section className="card">
      <div className="text-[10px] uppercase tracking-[.16em] text-purple-200/70">Research implementation status</div><h3 className="mt-1 text-sm font-semibold text-slate-100">CN objectives and evidence limits</h3>
      <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {[
          ["Split-point optimization", "Implemented · checkpoint matched", "badge-green", "The edge sends cumulative ResNet feature levels. The cloud verifies the checkpoint fingerprint and input normalization, runs the remaining stages, builds the FPN, then runs RPN and RoI heads. This verifies the software path, not an optimal split under real hospital networks."],
          ["Feature compression", "INT8 · measured request bytes", "badge-green", "INT8 values are transmitted as raw one-byte values with scale metadata. The reported ratio uses the complete serialized request body, including its header. It is a byte-count comparison, not a latency or clinical-utility result."],
          ["Learned compression", "Encoder/decoder integrated · checkpoint required", "badge-orange", "A trainable per-level bottleneck codec is available at edge and cloud. Both sides require the same codec file and matching trained backbone. A codec training script is provided, but rate-distortion and detection quality must be validated before research claims; no trained codec is bundled."],
          ["Federated learning", "Flower simulation + distributed client mode", "badge-orange", "The simulator uses LUNA16 subsets and labels results as simulation. Separate server/client processes support networked experiments with TLS server authentication, but clients are not certificate-authenticated; this client expects LUNA16 .mhd files. No hospitals have participated and no multi-institution validation is claimed."],
          ["Adaptive scheduling", "Integrated · local fallback", "badge-green", "Each scan probes the cloud before feature extraction and uses the local fallback below 5 Mbps or above 500 ms RTT. Cloud inference errors also select local fallback. Events are best-effort: if the backend is unreachable, it cannot record the event remotely."],
          ["Differential privacy", "Patient feature Gaussian mechanism · bounded scope", "badge-orange", "Live releases use joint per-scan L2 clipping, OS-backed Gaussian randomness, and strict basic (ε, δ) composition in a persistent accountant ledger. The conditional guarantee covers numeric feature values only; identifiers/traffic metadata, model training, and clinical or legal certification are outside scope. The offline utility experiment uses actual cloud continuation and generated pseudo-mask targets."],
        ].map(([title, state, badgeClass, detail]) => <article key={title} className="rounded-xl border border-slate-700/50 bg-slate-950/35 p-4"><div className="flex flex-wrap items-center justify-between gap-2"><h4 className="text-xs font-semibold text-slate-200">{title}</h4><span className={`badge ${badgeClass}`}>{state}</span></div><p className="mt-2 text-[11px] leading-relaxed text-slate-500">{detail}</p></article>)}
      </div>
    </section>

    <footer className="text-[10px] text-slate-600">Network probe RTT, edge processing, and cloud inference are separate component timings. Total pipeline time is shown only when measured on the edge; one-way transfer time is not inferred by subtracting or summing these values. No theoretical tensor sizes are presented as wire measurements.</footer>
  </div>;
}
