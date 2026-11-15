import React, { useState } from "react";
import { Routes, Route, NavLink, useLocation } from "react-router-dom";
import {
  LayoutDashboard, FlaskConical, BarChart3, Database,
  Scan, Network, Menu, X, Activity,
} from "lucide-react";
import Dashboard     from "./pages/Dashboard.jsx";
import Experiments   from "./pages/Experiments.jsx";
import Leaderboard   from "./pages/Leaderboard.jsx";
import InferencePage from "./pages/InferencePage.jsx";
import EdgeNetwork   from "./pages/EdgeNetwork.jsx";
import DatasetModel  from "./pages/DatasetModel.jsx";
import clsx from "clsx";

const NAV = [
  { to: "/",             label: "Dashboard",    icon: LayoutDashboard },
  { to: "/experiments",  label: "Experiments",  icon: FlaskConical    },
  { to: "/leaderboard",  label: "Leaderboard",  icon: BarChart3       },
  { to: "/inference",    label: "Inference",    icon: Scan            },
  { to: "/edge",         label: "Edge Network", icon: Network         },
  { to: "/dataset-model",label: "Dataset & Models", icon: Database    },
];

function Sidebar({ open, setOpen }) {
  return (
    <>
      {/* Overlay */}
      {open && (
        <div
          className="fixed inset-0 z-20 bg-black/60 lg:hidden"
          onClick={() => setOpen(false)}
        />
      )}
      <aside
        className={clsx(
          "sidebar-glass fixed top-0 left-0 z-30 h-full w-64 border-r",
          "flex flex-col transition-transform duration-200 ease-in-out",
          open ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
        )}
      >
        {/* Logo */}
        <div className="brand-header flex items-center gap-3 px-5 py-4 border-b">
          <span className="brand-mark text-2xl">🫁</span>
          <div>
            <div className="text-sm font-bold text-white leading-tight">Lung Nodule</div>
            <div className="text-xs text-slate-400">Mask R-CNN Pipeline</div>
          </div>
          <button
            className="ml-auto text-slate-400 hover:text-white lg:hidden"
            onClick={() => setOpen(false)}
          >
            <X size={18} />
          </button>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              onClick={() => setOpen(false)}
              className={({ isActive }) =>
                clsx(
                  "nav-link flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors",
                  isActive
                    ? "bg-cyan-500/15 text-cyan-400 border border-cyan-500/25"
                    : "text-slate-400 hover:text-slate-100 hover:bg-slate-800"
                )
              }
            >
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="sidebar-footer px-5 py-3 border-t">
          <div className="flex items-center gap-2">
            <Activity size={12} className="text-green-400" />
            <span className="text-xs text-slate-400">LUNA16 · ResNet-50-FPN</span>
          </div>
        </div>
      </aside>
    </>
  );
}

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const location = useLocation();

  const pageTitle = NAV.find(
    (n) => n.to === location.pathname || (n.to !== "/" && location.pathname.startsWith(n.to))
  )?.label ?? "Dashboard";

  return (
    <div className="app-shell min-h-screen flex">
      <Sidebar open={sidebarOpen} setOpen={setSidebarOpen} />

      {/* Main area */}
      <div className="flex-1 flex flex-col lg:ml-64 min-h-screen">
        {/* Top bar */}
        <header className="topbar-glass sticky top-0 z-10 flex items-center gap-4 px-5 py-3 border-b">
          <button
            className="text-slate-400 hover:text-white lg:hidden"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu size={20} />
          </button>
          <h1 className="text-sm font-semibold text-white">{pageTitle}</h1>
          <div className="ml-auto flex items-center gap-2">
            <span className="badge badge-cyan">DIP · Loss Weights</span>
            <span className="badge badge-purple">CN · Edge–Cloud</span>
          </div>
        </header>

        {/* Page content */}
        <main className="main-stage flex-1 p-5 md:p-6 overflow-auto">
          <Routes>
            <Route path="/"            element={<Dashboard />}     />
            <Route path="/experiments/*" element={<Experiments />} />
            <Route path="/leaderboard" element={<Leaderboard />}   />
            <Route path="/inference"   element={<InferencePage />} />
            <Route path="/edge"        element={<EdgeNetwork />}   />
            <Route path="/dataset-model" element={<DatasetModel />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
