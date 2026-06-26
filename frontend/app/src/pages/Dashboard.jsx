import { useEffect, useState, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";

import {
  FaBroadcastTower, FaShieldAlt, FaExclamationTriangle,
  FaChartLine, FaBell, FaBolt, FaHistory, FaWaveSquare,
  FaCircle, FaHardHat, FaUserShield, FaSkullCrossbones,
  FaUsers, FaBoxOpen,
} from "react-icons/fa";

import PageLayout from "../components/PageLayout";
import { getAnalyticsSummary, getAlerts, getCameraStatus, getWorkerCompliance } from "../services/api";
import "./dashboard.css";

const REFRESH_INTERVAL = 5000;
const DEFAULT_CAMERA_COUNT = 4;

/* ─── tiny pulse dot ───────────────────────────────────── */
function PDot({ color = "#00ff9d" }) {
  return <span className="pdot" style={{ "--c": color }} />;
}

/* ─── risk helpers ─────────────────────────────────────── */
const RISK_CLS = {
  CRITICAL: "db__incident--critical",
  HIGH: "db__incident--high",
  MEDIUM: "db__incident--medium",
  LOW: "db__incident--low",
};
const rCls = (r) => RISK_CLS[(r || "LOW").toUpperCase()] || RISK_CLS.LOW;

/* ─── system status → color / label ───────────────────── */
const STATUS_COLOR = {
  ACTIVE: "#00ff9d",
  OPERATIONAL: "#00ff9d",
  DEGRADED: "#f59e0b",
  WARNING: "#f59e0b",
  DOWN: "#ef4444",
  ERROR: "#ef4444",
  OFFLINE: "#ef4444",
};
const statusColor = (s) => STATUS_COLOR[(s || "").toUpperCase()] || "#4a6880";

/* ─── PPE icon lookup (falls back to a generic box icon) ── */
const PPE_ICON = {
  HELMET: <FaHardHat />,
  HARDHAT: <FaHardHat />,
  VEST: <FaUserShield />,
  BOOTS: <FaUserShield />,
  GLOVES: <FaUserShield />,
};
const ppeIcon = (name) => PPE_ICON[(name || "").toUpperCase()] || <FaBoxOpen />;

/* ─── 0-100 compliance/risk score → color band ────────── */
function scoreColor(score) {
  if (score == null) return "#4a6880";
  if (score >= 80) return "#00ff9d";
  if (score >= 50) return "#f59e0b";
  return "#ef4444";
}

/* ─── normalize an alert's PPE/danger-zone fields ─────────
   getAlerts' exact response shape isn't guaranteed, so this
   checks a few plausible field names rather than assuming one. */
function incidentDetails(item) {
  const missing =
    item.missing_ppe ?? item.missing_items ?? item.violations ?? [];
  const score =
    item.compliance_score ?? item.confidence_score ?? item.score ?? null;
  const dangerZone = Boolean(
    item.danger_zone ?? item.in_danger_zone ?? item.restricted_zone
  );
  return {
    missing: Array.isArray(missing) ? missing : [],
    score: typeof score === "number" ? Math.round(score) : null,
    dangerZone,
  };
}

/* ─── relative time helper ────────────────────────────── */
function timeAgo(iso) {
  if (!iso) return "—";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "—";
  const diffSec = Math.max(0, Math.floor((Date.now() - then) / 1000));
  if (diffSec < 5) return "just now";
  if (diffSec < 60) return `${diffSec}s ago`;
  const m = Math.floor(diffSec / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  return `${h}h ago`;
}

export default function Dashboard() {
  const navigate = useNavigate();
  const [summary, setSummary] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [cameras, setCameras] = useState([]);
  const [workers, setWorkers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [, setClockTick] = useState(0); // forces re-render so "Xs ago" stays live

  const isFirstLoad = useRef(true);

  // ── data load ─────────────────────────────────────────
  const loadDashboard = useCallback(async () => {
    try {
      // Pull alerts for every monitored camera, not just camera 0 —
      // getAlerts(camId) requires a camId and previously always
      // defaulted to 0, so cameras 1..N were never represented.
      const camIds = Array.from({ length: DEFAULT_CAMERA_COUNT }, (_, i) => i);

      const [analytics, alertLists, cameraStatuses, workerCompliance] = await Promise.all([
        getAnalyticsSummary(),
        Promise.all(
          camIds.map((id) =>
            getAlerts(id).catch(() => [])
          )
        ),
        Promise.all(
          camIds.map((id) =>
            getCameraStatus(id)
              .then((status) => ({ id, ...status, ok: true }))
              .catch(() => ({ id, ok: false }))
          )
        ),
        // Per-worker PPE compliance — same site-wide call as analytics,
        // so a single failure here shouldn't take down the rest of the page.
        getWorkerCompliance().catch(() => []),
      ]);

      const merged = alertLists
        .flat()
        .filter(Boolean)
        .sort((a, b) => {
          const ta = new Date(a.timestamp || a.created_at || 0).getTime();
          const tb = new Date(b.timestamp || b.created_at || 0).getTime();
          return tb - ta;
        });

      setSummary(analytics);
      setAlerts(merged);
      setCameras(cameraStatuses);
      setWorkers(Array.isArray(workerCompliance) ? workerCompliance : []);
      setError(null);
    } catch (err) {
      console.error("[Dashboard]", err);
      setError("Unable to load dashboard.");
    } finally {
      setLoading(false);
      isFirstLoad.current = false;
    }
  }, []);

  useEffect(() => {
    loadDashboard();
    const iv = setInterval(loadDashboard, REFRESH_INTERVAL);
    return () => clearInterval(iv);
  }, [loadDashboard]);

  // live-updating "time ago" ticker, independent of data refresh
  useEffect(() => {
    const iv = setInterval(() => setClockTick((t) => t + 1), 1000);
    return () => clearInterval(iv);
  }, []);

  // ── quick links (unchanged) ──────────────────────────
  const quickLinks = [
    { title: "Analytics", icon: <FaWaveSquare />, route: "/analytics" },
    { title: "Alerts", icon: <FaBell />, route: "/alerts" },
    { title: "History", icon: <FaHistory />, route: "/history" },
  ];

  // ── KPI cards (existing four, unchanged) ─────────────
  const kpis = [
    { label: "Live Cameras", value: summary?.active_cameras ?? DEFAULT_CAMERA_COUNT, icon: <FaBroadcastTower />, mod: "db__kpi-card--b" },
    { label: "Critical Alerts", value: summary?.high ?? 0, icon: <FaExclamationTriangle />, mod: "db__kpi-card--r" },
    { label: "Safety Score", value: `${summary?.safety_score ?? 98}%`, icon: <FaShieldAlt />, mod: "db__kpi-card--g" },
    { label: "Total Events", value: summary?.total ?? 0, icon: <FaChartLine />, mod: "db__kpi-card--y" },
  ];

  // ── risk distribution, from real summary fields ──────
  const riskTotal = (summary?.low ?? 0) + (summary?.medium ?? 0) + (summary?.high ?? 0);
  const riskRows = [
    { label: "HIGH", value: summary?.high ?? 0, color: "var(--r)" },
    { label: "MEDIUM", value: summary?.medium ?? 0, color: "var(--y)" },
    { label: "LOW", value: summary?.low ?? 0, color: "var(--g)" },
  ];

  // ── telemetry, from real summary fields only ─────────
  const telemetry = [
    { label: "System Status", val: summary?.system_status ?? "UNKNOWN" },
    { label: "Active Connections", val: summary?.active_connections ?? 0 },
    { label: "Events Sent", val: summary?.events_sent ?? 0 },
    { label: "Last Updated", val: timeAgo(summary?.last_updated) },
  ];

  // ── loading state ─────────────────────────────────────
  if (loading) {
    return (
      <PageLayout>
        <div className="db">
          <div className="db__kpi">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="db__kpi-card db__skeleton" />
            ))}
          </div>
          <div className="db__main">
            <div className="db__panel db__skeleton" style={{ minHeight: 260 }} />
            <div className="db__panel db__skeleton" style={{ minHeight: 260 }} />
          </div>
        </div>
      </PageLayout>
    );
  }

  // ── error state ───────────────────────────────────────
  if (error) {
    return (
      <PageLayout>
        <div className="db">
          <div className="db__empty">
            {error}
            <div>
              <motion.button
                className="db__nav-btn"
                whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }}
                onClick={loadDashboard}
              >
                Retry
              </motion.button>
            </div>
          </div>
        </div>
      </PageLayout>
    );
  }

  const sysColor = statusColor(summary?.system_status);

  return (
    <PageLayout>
      <div className="db">

        {/* ── HEADER ── */}
        <motion.div className="db__hdr" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <div>
            <div className="db__eyebrow"><PDot /> INFRAGUARD PLATFORM</div>
            <h1 className="db__h1">Command<span className="db__accent"> Center</span></h1>
            <p className="db__sub">
              Enterprise AI operational intelligence · Updated {timeAgo(summary?.last_updated)}
            </p>
          </div>
          <div className="db__status" style={{ color: sysColor, borderColor: `${sysColor}33`, background: `${sysColor}14` }}>
            <PDot color={sysColor} /> {(summary?.system_status || "UNKNOWN").toString().toUpperCase()}
          </div>
        </motion.div>

        {/* ── QUICK NAV (same routes) ── */}
        <div className="db__nav">
          {quickLinks.map((item) => (
            <motion.button
              key={item.route}
              className="db__nav-btn"
              whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }}
              onClick={() => navigate(item.route)}
            >
              {item.icon} {item.title}
            </motion.button>
          ))}
        </div>

        {/* ── KPI GRID ── */}
        <div className="db__kpi">
          {kpis.map((k, i) => (
            <motion.div
              key={k.label}
              className={`db__kpi-card ${k.mod}`}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.07 }}
              whileHover={{ y: -4 }}
            >
              <div className="db__kpi-label">{k.label}</div>
              <div className="db__kpi-val">{k.value}</div>
              <div className="db__kpi-icon">{k.icon}</div>
            </motion.div>
          ))}
        </div>

        {/* ── MAIN GRID ── */}
        <div className="db__main">

          {/* INCIDENT FEED */}
          <motion.div
            className="db__panel"
            initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
          >
            <div className="db__panel-title"><FaBell /> Realtime Incident Feed</div>
            <div className="db__feed">
              {alerts.length > 0
                ? alerts.slice(0, 8).map((item, i) => {
                  const { missing, score, dangerZone } = incidentDetails(item);
                  return (
                    <motion.div
                      key={item.id ?? `${item.camera_id ?? "c"}-${i}`}
                      className={`db__incident ${rCls(item.risk_level)}`}
                      initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.04 }}
                    >
                      <div className="db__incident-left">
                        <div className="db__incident-dot" />
                        <div>
                          <div className="db__incident-type">{item.event_type}</div>
                          <div className="db__incident-desc">{item.description}</div>

                          {missing.length > 0 && (
                            <div className="db__incident-reason">
                              <span className="db__incident-reason-label">Reason</span>
                              <div className="db__incident-reason-tags">
                                {missing.map((m, mi) => (
                                  <span key={mi} className="db__ppe-tag db__ppe-tag--missing">
                                    {ppeIcon(m)} {m} Missing
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}

                          {dangerZone && (
                            <div className="db__incident-danger">
                              <FaSkullCrossbones /> Danger Zone
                            </div>
                          )}
                        </div>
                      </div>

                      <div className="db__incident-right">
                        <div className="db__incident-risk">{item.risk_level}</div>
                        {score != null && (
                          <div
                            className="db__incident-score"
                            style={{ color: scoreColor(score) }}
                          >
                            {score}<span className="db__incident-score-max">/100</span>
                          </div>
                        )}
                      </div>
                    </motion.div>
                  );
                })
                : (
                  <div className="db__empty">
                    <PDot />
                    <div>No Active Incidents</div>
                    <small>All monitored construction zones are operating normally.</small>
                  </div>
                )
              }
            </div>
          </motion.div>

          {/* RIGHT COLUMN: telemetry + risk + cameras stacked */}
          <div className="db__side">

            {/* TELEMETRY — now wired to real summary fields */}
            <motion.div
              className="db__panel"
              initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
            >
              <div className="db__panel-title"><FaBolt /> System Telemetry</div>
              <div className="db__telem-grid">
                {telemetry.map((t) => (
                  <div key={t.label} className="db__telem-cell">
                    <div className="db__telem-label">{t.label}</div>
                    <div className="db__telem-val">{t.val}</div>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* RISK DISTRIBUTION — real low/medium/high split */}
            <motion.div
              className="db__panel"
              initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.15 }}
            >
              <div className="db__panel-title"><FaExclamationTriangle /> Risk Distribution</div>
              <div className="db__risk-list">
                {riskRows.map((r) => {
                  const pct = riskTotal > 0 ? Math.round((r.value / riskTotal) * 100) : 0;
                  return (
                    <div key={r.label} className="db__risk-row">
                      <div className="db__risk-row-top">
                        <span>{r.label}</span>
                        <span>{r.value}</span>
                      </div>
                      <div className="db__risk-bar-track">
                        <div
                          className="db__risk-bar-fill"
                          style={{ width: `${pct}%`, background: r.color }}
                        />
                      </div>
                    </div>
                  );
                })}
                {riskTotal === 0 && (
                  <div className="db__empty" style={{ padding: "14px 0" }}>
                    No events recorded yet.
                  </div>
                )}
              </div>
            </motion.div>

            {/* CAMERA STATUS — real per-camera calls to getCameraStatus */}
            <motion.div
              className="db__panel"
              initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <div className="db__panel-title"><FaBroadcastTower /> Camera Status</div>
              <div className="db__cam-grid">
                {cameras.map((cam) => {
                  // getCameraStatus's exact response shape isn't guaranteed,
                  // so we check common field names but don't assume "online"
                  // when none of them are present — unreachable is the only
                  // thing we can say for certain from a failed call.
                  let state = "unreachable";
                  if (cam.ok) {
                    if (cam.running === true || cam.active === true || cam.status === "running") {
                      state = "online";
                    } else if (cam.running === false || cam.active === false || cam.status === "stopped") {
                      state = "idle";
                    } else {
                      state = "reachable";
                    }
                  }
                  const dotColor = state === "online" ? "#00ff9d" : state === "unreachable" ? "#ef4444" : "#4a6880";
                  return (
                    <div key={cam.id} className="db__cam-cell">
                      <FaCircle style={{ color: dotColor, fontSize: "0.5rem" }} />
                      <span>Camera {cam.id}</span>
                      <span className="db__cam-state">{state}</span>
                    </div>
                  );
                })}
              </div>
            </motion.div>

            {/* WORKER COMPLIANCE — per-worker PPE compliance breakdown */}
            <motion.div
              className="db__panel"
              initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.25 }}
            >
              <div className="db__panel-title"><FaUsers /> Worker Compliance</div>
              <div className="db__worker-grid">
                {workers.length > 0
                  ? workers.map((w, i) => {
                    const id = w.worker_id ?? w.id ?? i;
                    const compliance =
                      typeof w.compliance === "number"
                        ? Math.round(w.compliance)
                        : typeof w.compliance_score === "number"
                          ? Math.round(w.compliance_score)
                          : null;
                    const risk = (w.risk_level || w.risk || "UNKNOWN").toString().toUpperCase();
                    const missing = Array.isArray(w.missing_ppe ?? w.missing_items)
                      ? (w.missing_ppe ?? w.missing_items)
                      : [];
                    return (
                      <div key={id} className={`db__worker-card ${rCls(risk)}`}>
                        <div className="db__worker-card-hdr">
                          <span className="db__worker-name">Worker {id}</span>
                          <span className="db__incident-risk">{risk}</span>
                        </div>

                        <div className="db__worker-compliance">
                          <div className="db__worker-compliance-top">
                            <span>Compliance</span>
                            <span style={{ color: scoreColor(compliance) }}>
                              {compliance != null ? `${compliance}%` : "—"}
                            </span>
                          </div>
                          <div className="db__risk-bar-track">
                            <div
                              className="db__risk-bar-fill"
                              style={{
                                width: `${compliance ?? 0}%`,
                                background: scoreColor(compliance),
                              }}
                            />
                          </div>
                        </div>

                        {missing.length > 0 && (
                          <div className="db__incident-reason">
                            <span className="db__incident-reason-label">Missing</span>
                            <div className="db__incident-reason-tags">
                              {missing.map((m, mi) => (
                                <span key={mi} className="db__ppe-tag db__ppe-tag--missing">
                                  {ppeIcon(m)} {m}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })
                  : (
                    <div className="db__empty" style={{ padding: "14px 0" }}>
                      No worker compliance data available.
                    </div>
                  )
                }
              </div>
            </motion.div>

          </div>

        </div>
      </div>
    </PageLayout>
  );
}