import axios from "axios";

const BASE = "/api";

const http = axios.create({
  baseURL: BASE,
  headers: { "Content-Type": "application/json" },
  timeout: 60000,
});

http.interceptors.response.use(
  (r) => r.data,
  (err) => {
    const msg = err.response?.data?.detail || err.message || "Request failed";
    return Promise.reject(new Error(msg));
  }
);

// --- Experiments ---
export const api = {
  // Dashboard stats
  getStats: ()                    => http.get("/results/stats"),
  getInfo:  ()                    => http.get("/info"),

  // Experiments CRUD
  listExperiments: (params = {})  => http.get("/experiments/", { params }),
  getExperiment:   (id)           => http.get(`/experiments/${id}`),
  createExperiment:(data)         => http.post("/experiments/", data),
  startExperiment: (id)           => http.post(`/experiments/${id}/start`),
  deleteExperiment:(id)           => http.delete(`/experiments/${id}`),
  getRuns:         (id)           => http.get(`/experiments/${id}/runs`),
  getExperimentLogs:(id)          => http.get(`/experiments/${id}/logs`),
  getSummary:      (id)           => http.get(`/experiments/${id}/summary`),
  getLearningCurve:(id)           => http.get(`/results/experiment/${id}/learning-curve`),

  // Leaderboard
  getLeaderboard:  ()             => http.get("/results/leaderboard"),
  getPareto:       ()             => http.get("/results/pareto"),

  // Inference
  predictUpload:  (formData)      => http.post("/inference/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  }),
  predictBase64:  (data)          => http.post("/inference/predict", data),
  listInferences: (params = {})   => http.get("/inference/results", { params }),

  // Edge / Network
  listEdgeNodes:        ()        => http.get("/edge/nodes"),
  getNetworkLogs:  (params = {})  => http.get("/edge/network-log", { params }),
  getNetworkStats:      ()        => http.get("/edge/network-stats"),
  getLatencyAnalysis:   ()        => http.get("/edge/split-latency-analysis"),
};

// WebSocket helper for live training progress
export function openTrainingWS(experimentId, onMessage, onClose) {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const host  = window.location.host;
  const ws    = new WebSocket(`${proto}://${host}/api/experiments/${experimentId}/ws`);
  ws.onmessage = (e) => {
    try { onMessage(JSON.parse(e.data)); } catch {}
  };
  ws.onclose   = onClose;
  ws.onerror   = (e) => console.error("WS error", e);
  return ws;
}
