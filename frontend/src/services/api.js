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
  getDashboardHealth:   ()         => http.get("/dashboard/health"),
  getDashboardSummary:  ()         => http.get("/dashboard/summary"),
  getDashboardActivity: (params={})=> http.get("/dashboard/activity", { params }),
  getDatasetStatus:     ()         => http.get("/dashboard/dataset"),
  getSystemMetrics:     ()         => http.get("/dashboard/system-metrics"),

  // Experiments CRUD
  listExperiments: (params = {})  => http.get("/experiments/", { params }),
  getExperiment:   (id)           => http.get(`/experiments/${id}`),
  getExperimentDetails:(id)       => http.get(`/experiments/${id}/details`),
  createExperiment:(data)         => http.post("/experiments/", data),
  startExperiment: (id)           => http.post(`/experiments/${id}/start`),
  deleteExperiment:(id)           => http.delete(`/experiments/${id}`),
  getRuns:         (id)           => http.get(`/experiments/${id}/runs`),
  getExperimentLogs:(id)          => http.get(`/experiments/${id}/logs`),
  getSummary:      (id)           => http.get(`/experiments/${id}/summary`),
  getLearningCurve:(id)           => http.get(`/results/experiment/${id}/learning-curve`),

  // Leaderboard
  getLeaderboard:  (params = {})  => http.get("/results/leaderboard", { params }),
  getPareto:       (params = {})  => http.get("/results/pareto", { params }),

  // Inference
  getInferenceStatus:()           => http.get("/inference/status"),
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
export function openTrainingWS(experimentId, onMessage, onClose, onOpen) {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const host  = window.location.host;
  const ws    = new WebSocket(`${proto}://${host}/api/experiments/${experimentId}/ws`);
  ws.onmessage = (e) => {
    try { onMessage(JSON.parse(e.data)); } catch { /* Ignore malformed progress frames. */ }
  };
  ws.onopen = onOpen;
  ws.onclose   = onClose;
  ws.onerror   = (e) => console.error("WS error", e);
  return ws;
}
