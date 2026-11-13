import React, { useEffect, useState } from "react";
import { api } from "../services/api.js";
import { Loader2, TrendingUp, Star } from "lucide-react";
import {
  ScatterChart, Scatter, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, ReferenceLine, Legend,
} from "recharts";

const COLS = [
  { k: "rank",             label: "#",            fmt: (v) => v },
  { k: "config_id",        label: "Config",       fmt: (v) => <span className="font-mono text-cyan-400 text-xs">{v}</span> },
  { k: "loss_description", label: "Loss Formula", fmt: (v) => <span className="text-xs">{v}</span> },
  { k: "alpha",            label: "α",            fmt: (v) => v?.toFixed(1) },
  { k: "beta",             label: "β",            fmt: (v) => v?.toFixed(1) },
  { k: "gamma",            label: "γ",            fmt: (v) => v?.toFixed(1) },
  { k: "best_dice",        label: "Dice ↑",       fmt: (v) => <span className="text-green-400 font-semibold">{v?.toFixed(4) ?? "—"}</span> },
  { k: "best_hd95",        label: "HD95 ↓",       fmt: (v) => <span className="text-orange-400">{v?.toFixed(2) ?? "—"}</span> },
  { k: "best_boundary_f1", label: "BdryF1 ↑",    fmt: (v) => <span className="text-purple-400">{v?.toFixed(4) ?? "—"}</span> },
];

const TooltipContent = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg p-3 text-xs">
      <div className="font-mono text-cyan-400 mb-1">{d?.config_id}</div>
      <div className="text-slate-300">{d?.loss_description}</div>
      <div className="mt-1 space-y-0.5">
        <div>Dice: <span className="text-green-400 font-semibold">{d?.best_dice?.toFixed(4)}</span></div>
        <div>HD95: <span className="text-orange-400">{d?.best_hd95?.toFixed(2)}</span></div>
      </div>
    </div>
  );
};

export default function Leaderboard() {
  const [lb, setLB]         = useState([]);
  const [pareto, setPareto] = useState([]);
  const [loading, setLoad]  = useState(true);
  const [sortKey, setSort]  = useState("rank");
  const [sortAsc, setSortAsc] = useState(true);

  useEffect(() => {
    setLoad(true);
    Promise.all([api.getLeaderboard(), api.getPareto()])
      .then(([l, p]) => { setLB(l); setPareto(p); })
      .finally(() => setLoad(false));
  }, []);

  const sorted = [...lb].sort((a, b) => {
    const av = a[sortKey] ?? 0, bv = b[sortKey] ?? 0;
    return sortAsc ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1);
  });

  const toggleSort = (k) => {
    if (k === sortKey) setSortAsc((a) => !a);
    else { setSort(k); setSortAsc(true); }
  };

  const paretoIds = new Set(pareto.map((p) => p.experiment_id));

  // Scatter data: Dice (x) vs HD95 (y)
  const scatterAll    = lb.filter(r => r.best_dice != null && r.best_hd95 != null)
    .map(r => ({ ...r, x: r.best_dice, y: r.best_hd95 }));
  const scatterPareto = pareto.filter(r => r.best_dice != null && r.best_hd95 != null)
    .map(r => ({ ...r, x: r.best_dice, y: r.best_hd95 }));

  if (loading) return (
    <div className="flex justify-center py-16"><Loader2 className="animate-spin text-cyan-400" size={28} /></div>
  );

  return (
    <div className="space-y-6">
      {/* Pareto Banner */}
      {pareto.length > 0 && (
        <div className="card border-cyan-500/30 bg-cyan-500/5">
          <div className="flex items-center gap-2 mb-3">
            <Star size={16} className="text-cyan-400" />
            <span className="text-sm font-bold text-white">Pareto Frontier ({pareto.length} configs)</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {pareto.map((p) => (
              <div key={p.experiment_id}
                   className="bg-slate-800 rounded-lg px-3 py-1.5 border border-cyan-500/20 text-xs">
                <span className="font-mono text-cyan-400">{p.config_id}</span>
                <span className="text-slate-400 ml-2">Dice={p.best_dice?.toFixed(3)} HD95={p.best_hd95?.toFixed(1)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Scatter: Dice vs HD95 */}
      <div className="card">
        <h2 className="text-sm font-bold text-white mb-1">Dice ↑ vs HD95 ↓ — Pareto Space</h2>
        <p className="text-xs text-slate-400 mb-4">Top-right + bottom-right quadrant = best balance. ★ = Pareto optimal.</p>
        <ResponsiveContainer width="100%" height={280}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
            <XAxis dataKey="x" name="Dice"  type="number" domain={[0, 1]}
                   label={{ value: "Dice ↑", position: "insideBottom", fill: "#64748b", fontSize: 11 }}
                   tick={{ fill: "#64748b", fontSize: 11 }} />
            <YAxis dataKey="y" name="HD95" type="number"
                   label={{ value: "HD95 ↓", angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 11 }}
                   tick={{ fill: "#64748b", fontSize: 11 }} />
            <Tooltip content={<TooltipContent />} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Scatter name="All experiments" data={scatterAll}    fill="#475569" opacity={0.7} />
            <Scatter name="Pareto optimal"  data={scatterPareto} fill="#06b6d4" opacity={1}   />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Sortable Table */}
      <div className="card">
        <h2 className="text-sm font-bold text-white mb-4">
          Full Leaderboard ({lb.length} experiments)
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-800">
                {COLS.map(({ k, label }) => (
                  <th
                    key={k}
                    className="text-left py-2 px-3 text-xs text-slate-400 uppercase tracking-wider font-semibold cursor-pointer select-none hover:text-white"
                    onClick={() => toggleSort(k)}
                  >
                    {label} {sortKey === k ? (sortAsc ? "↑" : "↓") : ""}
                  </th>
                ))}
                <th className="py-2 px-3 text-xs text-slate-400">Pareto</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((r) => (
                <tr key={r.experiment_id}
                    className={`border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors
                      ${paretoIds.has(r.experiment_id) ? "bg-cyan-500/5" : ""}`}>
                  {COLS.map(({ k, fmt }) => (
                    <td key={k} className="py-2.5 px-3 text-slate-300">
                      {fmt(r[k])}
                    </td>
                  ))}
                  <td className="py-2.5 px-3">
                    {paretoIds.has(r.experiment_id) && (
                      <Star size={14} className="text-cyan-400" />
                    )}
                  </td>
                </tr>
              ))}
              {lb.length === 0 && (
                <tr><td colSpan={10} className="py-10 text-center text-slate-500">
                  No completed experiments yet.
                </td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
