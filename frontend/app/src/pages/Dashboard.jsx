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

  // ── data load (unchanged logic) ──────────────────────
  useEffect(() => {
    loadDashboard();
    const iv = setInterval(loadDashboard, 5000);
    return () => clearInterval(iv);
  }, []);

  const loadDashboard = async () => {
    try {
      const analytics = await getAnalyticsSummary();
      const alertFeed = await getAlerts();
      setSummary(analytics);
      setAlerts(alertFeed || []);
    } catch (err) {
      console.error("[DASHBOARD ERROR]", err);
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
    { label: "Live Cameras", value: "4", icon: <FaBroadcastTower />, mod: "db__kpi-card--b" },
    { label: "Critical Alerts", value: summary?.high ?? 0, icon: <FaExclamationTriangle />, mod: "db__kpi-card--r" },
    { label: "Safety Score", value: `${summary?.safety_score ?? 98}%`, icon: <FaShieldAlt />, mod: "db__kpi-card--g" },
    { label: "Total Events", value: summary?.total ?? 0, icon: <FaChartLine />, mod: "db__kpi-card--y" },
  ];

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
          {quickLinks.map((item, i) => (
            <motion.button
              key={i}
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
              key={i}
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
                    key={i}
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
                : <div className="db__empty">No incidents detected</div>
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
              ].map((t, i) => (
                <div key={i} className="db__telem-cell">
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