import React, { useEffect, useState } from "react";
import { api } from "../services/api.js";
import { Loader2, Wifi, Server, Zap, Database } from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid,
  ResponsiveContainer, Legend, LineChart, Line,
} from "recharts";

function NodeCard({ node }) {
  return (
    <div className={`card border ${node.is_online ? "border-green-500/30 bg-green-500/5" : "border-slate-700"}`}>
      <div className="flex items-start justify-between mb-3">
        <div>
          <div className="text-sm font-bold text-white">{node.node_id}</div>
          <div className="text-xs text-slate-400">{node.hostname} · {node.ip_address}</div>
        </div>
        <span className={`badge ${node.is_online ? "badge-green" : "badge-red"}`}>
          {node.is_online ? "Online" : "Offline"}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-3 text-xs">
        <div>
          <div className="text-slate-400 mb-0.5">Split Layer</div>
          <div className="font-mono text-cyan-400">{node.split_layer}</div>
        </div>
        <div>
          <div className="text-slate-400 mb-0.5">Compression</div>
          <div className="text-orange-400">{node.compression_ratio?.toFixed(2)}×</div>
        </div>
        <div>
          <div className="text-slate-400 mb-0.5">Avg Latency</div>
          <div className="text-yellow-400">{node.avg_latency_ms?.toFixed(1) ?? "—"} ms</div>
        </div>
        <div>
          <div className="text-slate-400 mb-0.5">Bandwidth</div>
          <div className="text-green-400">{node.avg_bandwidth_mbps?.toFixed(1) ?? "—"} Mbps</div>
        </div>
      </div>
      {node.last_seen && (
        <div className="text-xs text-slate-500 mt-2">
          Last seen: {new Date(node.last_seen).toLocaleTimeString()}
        </div>
      )}
    </div>
  );
}

export default function EdgeNetwork() {
  const [nodes, setNodes]             = useState([]);
  const [netStats, setNetStats]       = useState([]);
  const [latencyAnalysis, setLatency] = useState([]);
  const [netLogs, setNetLogs]         = useState([]);
  const [loading, setLoading]         = useState(true);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      api.listEdgeNodes(),
      api.getNetworkStats(),
      api.getLatencyAnalysis(),
      api.getNetworkLogs({ limit: 100 }),
    ])
      .then(([n, ns, la, nl]) => {
        setNodes(n);
        setNetStats(ns);
        setLatency(la);
        setNetLogs(nl);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  // Prepare split-point latency chart
  const splitOrder = ["layer1", "layer2", "layer3", "cloud_only"];
  const latencyChart = splitOrder
    .map(layer => latencyAnalysis.find(r => r.split_layer === layer))
    .filter(Boolean)
    .map(r => ({
      name: r.split_layer,
      "Total Latency (ms)": r.avg_total_latency_ms,
      "Payload (KB)": r.avg_payload_kb,
    }));

  // Compression vs Latency scatter for different methods
  const compChart = netStats.map(r => ({
    name: `${r.split_layer}/${r.compression_method}`,
    "Avg Latency (ms)": r.avg_latency_ms,
    "Compression ×": r.avg_compression_ratio,
    n: r.n_requests,
  }));

  // Rolling latency over time
  const timeChart = netLogs.slice(-50).map((l, i) => ({
    idx: i + 1,
    "Latency (ms)":  l.latency_ms,
    "Payload (KB)":  l.payload_size_kb,
    "Cloud Inf (ms)": l.inference_time_ms,
  }));

  if (loading) return (
    <div className="flex justify-center py-16">
      <Loader2 className="animate-spin text-cyan-400" size={28} />
    </div>
  );

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-bold text-white">Edge–Cloud Pipeline (CN Research)</h2>
        <p className="text-sm text-slate-400">Network metrics, split-point analysis, and compression tradeoffs</p>
      </div>

      {/* Edge nodes grid */}
      <div>
        <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
          <Wifi size={15} className="text-cyan-400" /> Edge Nodes
        </h3>
        {nodes.length === 0 ? (
          <div className="card text-center py-8 text-slate-500 text-sm">
            No edge nodes registered. Start the edge service to register automatically.
          </div>
        ) : (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {nodes.map((n) => <NodeCard key={n.id} node={n} />)}
          </div>
        )}
      </div>

      {/* CN1: Split-point latency analysis */}
      <div className="card">
        <h3 className="text-sm font-bold text-white mb-1 flex items-center gap-2">
          <Zap size={15} className="text-yellow-400" /> CN1 · Split-Point Latency Analysis
        </h3>
        <p className="text-xs text-slate-400 mb-4">
          Total latency (edge compute + network + cloud inference) vs payload size per split layer.
          Optimal split = lowest total latency for a given network condition.
        </p>
        {latencyChart.length === 0 ? (
          <div className="text-sm text-slate-500 text-center py-8">
            No data yet — submit scans via the edge service to populate this chart.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={latencyChart}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
              <XAxis dataKey="name" tick={{ fill: "#64748b", fontSize: 11 }} />
              <YAxis yAxisId="left"  tick={{ fill: "#64748b", fontSize: 11 }} />
              <YAxis yAxisId="right" orientation="right" tick={{ fill: "#64748b", fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar yAxisId="left"  dataKey="Total Latency (ms)" fill="#06b6d4" radius={[4,4,0,0]} />
              <Bar yAxisId="right" dataKey="Payload (KB)"       fill="#a78bfa" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* CN2: Compression analysis */}
      <div className="card">
        <h3 className="text-sm font-bold text-white mb-1 flex items-center gap-2">
          <Database size={15} className="text-purple-400" /> CN2 · Compression Method Comparison
        </h3>
        <p className="text-xs text-slate-400 mb-4">
          Compression ratio achieved per split layer and method. Higher ratio = less bandwidth.
          Research goal: &gt;50% reduction with &lt;2% Dice degradation.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-800">
                {["Split Layer","Method","Avg Latency (ms)","Payload (KB)","Compression ×","Requests"].map(h => (
                  <th key={h} className="text-left py-2 px-3 text-xs text-slate-400 uppercase tracking-wider">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {netStats.map((r, i) => (
                <tr key={i} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                  <td className="py-2 px-3 font-mono text-cyan-400 text-xs">{r.split_layer}</td>
                  <td className="py-2 px-3">
                    <span className="badge badge-purple">{r.compression_method}</span>
                  </td>
                  <td className="py-2 px-3 text-yellow-400">{r.avg_latency_ms}</td>
                  <td className="py-2 px-3 text-slate-300">{r.avg_payload_kb}</td>
                  <td className="py-2 px-3 text-green-400 font-semibold">{r.avg_compression_ratio}×</td>
                  <td className="py-2 px-3 text-slate-400">{r.n_requests}</td>
                </tr>
              ))}
              {netStats.length === 0 && (
                <tr><td colSpan={6} className="py-6 text-center text-slate-500 text-sm">No network data yet</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Live latency timeline */}
      {timeChart.length > 0 && (
        <div className="card">
          <h3 className="text-sm font-bold text-white mb-4 flex items-center gap-2">
            <Server size={15} className="text-green-400" /> Live Pipeline Latency Timeline
          </h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={timeChart}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
              <XAxis dataKey="idx" tick={{ fill: "#64748b", fontSize: 10 }} />
              <YAxis tick={{ fill: "#64748b", fontSize: 10 }} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="Latency (ms)"   stroke="#facc15" strokeWidth={1.5} dot={false} />
              <Line type="monotone" dataKey="Cloud Inf (ms)" stroke="#34d399" strokeWidth={1.5} dot={false} />
              <Line type="monotone" dataKey="Payload (KB)"   stroke="#818cf8" strokeWidth={1}   dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Research annotations */}
      <div className="grid md:grid-cols-3 gap-4">
        {[
          { label: "CN1 · Split Point", color: "cyan",   desc: "Optimal: layer2 for most hospital LANs (8× spatial reduction, 512ch features, ~32KB payload vs ~49KB raw patch)" },
          { label: "CN3 · Federated",   color: "purple", desc: "FedAvg across LUNA16 10 subsets. Connect your Flower server to run federated experiments across simulated silos." },
          { label: "CN5 · Privacy",     color: "green",  desc: "DP noise injection at split layer. ε < 5.0 targets privacy budget. Enable via DIFFERENTIAL_PRIVACY=true env var." },
        ].map(({ label, color, desc }) => (
          <div key={label} className={`card border-${color}-500/20 bg-${color}-500/5`}>
            <div className={`text-xs font-bold text-${color}-400 mb-2`}>{label}</div>
            <p className="text-xs text-slate-400">{desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
