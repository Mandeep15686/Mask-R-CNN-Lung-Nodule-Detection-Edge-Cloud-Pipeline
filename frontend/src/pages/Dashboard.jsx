import React, { useEffect, useState } from "react";
import { api } from "../services/api.js";
import { FlaskConical, Scan, TrendingUp, Layers, AlertCircle, Loader2 } from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
} from "recharts";

function StatCard({ label, value, icon: Icon, color = "cyan", sub }) {
  const colors = {
    cyan:   "text-cyan-400 bg-cyan-500/10 border-cyan-500/20",
    green:  "text-green-400 bg-green-500/10 border-green-500/20",
    purple: "text-purple-400 bg-purple-500/10 border-purple-500/20",
    orange: "text-orange-400 bg-orange-500/10 border-orange-500/20",
  };
  return (
    <div className="card stat-card flex items-start gap-4">
      <div className={`stat-icon p-2.5 rounded-lg border ${colors[color]}`}>
        <Icon size={18} className={colors[color].split(" ")[0]} />
      </div>
      <div>
        <div className="text-2xl font-bold text-white">{value ?? "—"}</div>
        <div className="text-sm text-slate-400">{label}</div>
        {sub && <div className="text-xs text-slate-500 mt-0.5">{sub}</div>}
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [stats, setStats]         = useState(null);
  const [leaderboard, setLB]      = useState([]);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([api.getStats(), api.getLeaderboard()])
      .then(([s, lb]) => { setStats(s); setLB(lb.slice(0, 5)); })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  // Prepare sparkline data from leaderboard
  const chartData = leaderboard.map((r, i) => ({
    name: r.config_id.slice(0, 10),
    Dice: r.best_dice ?? 0,
    HD95: r.best_hd95 ?? 0,
    BoundaryF1: r.best_boundary_f1 ?? 0,
  }));

  if (loading) return (
    <div className="flex items-center justify-center h-64">
      <Loader2 className="animate-spin text-cyan-400" size={32} />
    </div>
  );
  if (error) return (
    <div className="card flex items-center gap-3 text-red-400">
      <AlertCircle size={18} /> {error}
    </div>
  );

  return (
    <div className="space-y-6">
      <section className="dashboard-hero">
        <div className="hero-copy">
          <div className="eyebrow"><span className="live-dot" /> Clinical research workspace</div>
          <h2>Research <span>Overview</span></h2>
          <p>Model performance, experiment progress, and inference activity in one view.</p>
        </div>
        <div className="hero-aside">
          <div className="hero-orb"><Layers size={23} /></div>
          <div>
            <div className="hero-aside-label">PIPELINE</div>
            <div className="hero-aside-value">LUNA16 · 2.5D</div>
            <div className="hero-aside-sub">ResNet-50 with FPN</div>
          </div>
        </div>
      </section>

      {/* Stat Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Total Experiments" value={stats?.total_experiments} icon={FlaskConical} color="cyan" />
        <StatCard label="Running"           value={stats?.running}           icon={Layers}       color="orange" />
        <StatCard label="Completed"         value={stats?.done}              icon={TrendingUp}   color="green" />
        <StatCard label="Total Inferences"  value={stats?.total_inferences}  icon={Scan}         color="purple" />
      </div>

      {/* Best Dice Banner */}
      {stats?.best_dice_overall && (
        <div className="card border-cyan-500/30 bg-cyan-500/5">
          <div className="flex items-center gap-3">
            <TrendingUp size={20} className="text-cyan-400" />
            <div>
              <div className="text-xs text-slate-400 uppercase tracking-wider mb-0.5">Best Validation Dice — All Experiments</div>
              <div className="text-3xl font-bold text-cyan-400">{stats.best_dice_overall}</div>
            </div>
          </div>
        </div>
      )}

      {/* Top 5 Chart */}
      {chartData.length > 0 && (
        <div className="card">
          <h2 className="text-sm font-semibold text-white mb-4">
            Top 5 Experiments — Metric Comparison
          </h2>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
              <XAxis dataKey="name" tick={{ fill: "#64748b", fontSize: 11 }} />
              <YAxis tick={{ fill: "#64748b", fontSize: 11 }} domain={[0, 1]} />
              <Tooltip
                contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }}
                labelStyle={{ color: "#cbd5e1" }}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line type="monotone" dataKey="Dice"       stroke="#06b6d4" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="BoundaryF1" stroke="#a78bfa" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Top experiments table */}
      <div className="card">
        <h2 className="text-sm font-semibold text-white mb-4">Top Experiments by Dice</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-800">
                {["Rank","Config","Loss Formula","Dice","HD95","BdryF1","Status"].map(h => (
                  <th key={h} className="text-left py-2 px-3 text-xs text-slate-400 uppercase tracking-wider font-semibold">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {leaderboard.map((r) => (
                <tr key={r.experiment_id} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                  <td className="py-2.5 px-3 font-bold text-slate-400">#{r.rank}</td>
                  <td className="py-2.5 px-3 font-mono text-cyan-400 text-xs">{r.config_id}</td>
                  <td className="py-2.5 px-3 text-slate-300 text-xs">{r.loss_description}</td>
                  <td className="py-2.5 px-3 font-semibold text-green-400">{r.best_dice?.toFixed(4) ?? "—"}</td>
                  <td className="py-2.5 px-3 text-orange-400">{r.best_hd95?.toFixed(2) ?? "—"}</td>
                  <td className="py-2.5 px-3 text-purple-400">{r.best_boundary_f1?.toFixed(4) ?? "—"}</td>
                  <td className="py-2.5 px-3">
                    <StatusBadge status={r.status} />
                  </td>
                </tr>
              ))}
              {leaderboard.length === 0 && (
                <tr><td colSpan={7} className="py-8 text-center text-slate-500">No experiments yet — create one in Experiments</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export function StatusBadge({ status }) {
  const map = {
    pending: "badge-slate",
    running: "badge-orange",
    done:    "badge-green",
    failed:  "badge-red",
  };
  return <span className={`badge ${map[status] ?? "badge-slate"}`}>{status}</span>;
}
