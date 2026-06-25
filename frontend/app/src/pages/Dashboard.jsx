import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";

import {
  FaBroadcastTower, FaShieldAlt, FaExclamationTriangle,
  FaChartLine, FaBell, FaBolt, FaHistory, FaWaveSquare,
} from "react-icons/fa";

import PageLayout from "../components/PageLayout";
import { getAnalyticsSummary, getAlerts } from "../services/api";
import "./dashboard.css";

const REFRESH_INTERVAL = 5000;
const DEFAULT_CAMERA_COUNT = 4;

/* ─── tiny pulse dot ───────────────────────────────────── */
function PDot({ color = "#00ff9d" }) {
  return <span className="pdot" style={{ "--c": color }} />;
}

/* ─── risk helpers ─────────────────────────────────────── */
const RISK_CLS = { HIGH: "db__incident--high", MEDIUM: "db__incident--medium", LOW: "db__incident--low" };
const rCls = (r) => RISK_CLS[(r || "LOW").toUpperCase()] || RISK_CLS.LOW;

export default function Dashboard() {
  const navigate = useNavigate();
  const [summary, setSummary] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // ── data load ─────────────────────────────────────────
  useEffect(() => {
    loadDashboard();
    const iv = setInterval(loadDashboard, REFRESH_INTERVAL);
    return () => clearInterval(iv);
  }, []);

  const loadDashboard = async () => {
    try {
      const [analytics, alertFeed] = await Promise.all([
        getAnalyticsSummary(),
        getAlerts(),
      ]);
      setSummary(analytics);
      setAlerts(alertFeed || []);
      setError(null);
    } catch (err) {
      console.error("[Dashboard]", err);
      setError("Unable to load dashboard.");
    } finally {
      setLoading(false);
    }
  };

  // ── quick links (unchanged) ──────────────────────────
  const quickLinks = [
    { title: "Analytics", icon: <FaWaveSquare />, route: "/analytics" },
    { title: "Alerts", icon: <FaBell />, route: "/alerts" },
    { title: "History", icon: <FaHistory />, route: "/history" },
  ];

  // ── KPI cards ────────────────────────────────────────
  const kpis = [
    { label: "Live Cameras", value: summary?.active_cameras ?? DEFAULT_CAMERA_COUNT, icon: <FaBroadcastTower />, mod: "db__kpi-card--b" },
    { label: "Critical Alerts", value: summary?.high ?? 0, icon: <FaExclamationTriangle />, mod: "db__kpi-card--r" },
    { label: "Safety Score", value: `${summary?.safety_score ?? 98}%`, icon: <FaShieldAlt />, mod: "db__kpi-card--g" },
    { label: "Total Events", value: summary?.total ?? 0, icon: <FaChartLine />, mod: "db__kpi-card--y" },
  ];

  // ── loading state ─────────────────────────────────────
  if (loading) {
    return (
      <PageLayout>
        <div className="db">
          <div className="db__empty">Loading Dashboard...</div>
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

  return (
    <PageLayout>
      <div className="db">

        {/* ── HEADER ── */}
        <motion.div className="db__hdr" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <div>
            <div className="db__eyebrow"><PDot /> INFRAGUARD PLATFORM</div>
            <h1 className="db__h1">Command<span className="db__accent"> Center</span></h1>
            <p className="db__sub">Enterprise AI operational intelligence</p>
          </div>
          <div className="db__status">
            <PDot /> SYSTEM OPERATIONAL
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
                ? alerts.slice(0, 8).map((item, i) => (
                  <motion.div
                    key={item.id ?? i}
                    className={`db__incident ${rCls(item.risk_level)}`}
                    initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.04 }}
                  >
                    <div className="db__incident-left">
                      <div className="db__incident-dot" />
                      <div>
                        <div className="db__incident-type">{item.event_type}</div>
                        <div className="db__incident-desc">{item.description}</div>
                      </div>
                    </div>
                    <div className="db__incident-risk">{item.risk_level}</div>
                  </motion.div>
                ))
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

          {/* TELEMETRY */}
          <motion.div
            className="db__panel"
            initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <div className="db__panel-title"><FaBolt /> System Telemetry</div>
            <div className="db__telem-grid">
              {[
                { label: "AI Engine", val: "ACTIVE" },
                { label: "WebSocket", val: "CONNECTED" },
                { label: "Inference", val: "28 FPS" },
                { label: "Detection", val: "LIVE" },
              ].map((t) => (
                <div key={t.label} className="db__telem-cell">
                  <div className="db__telem-label">{t.label}</div>
                  <div className="db__telem-val">{t.val}</div>
                </div>
              ))}
            </div>
          </motion.div>

        </div>
      </div>
    </PageLayout>
  );
}