import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  FaShieldAlt, FaExclamationTriangle, FaBroadcastTower,
  FaChartLine, FaBolt, FaCheckCircle, FaServer, FaWaveSquare,
} from "react-icons/fa";

import PageLayout from "../components/PageLayout";
import { getAnalyticsSummary, getAlerts } from "../services/api";
import "./analytics.css";

function PDot({ color = "#00ff9d" }) {
  return <span className="pdot" style={{ "--c": color }} />;
}

const INC_CLS = { HIGH: "an__incident--high", MEDIUM: "an__incident--medium", LOW: "an__incident--low" };
const incCls = (r) => INC_CLS[(r || "LOW").toUpperCase()] || INC_CLS.LOW;

export default function AnalyticsDashboard() {
  const [summary, setSummary] = useState(null);
  const [alerts, setAlerts] = useState([]);

  /* ── load (unchanged) ──────────────────────────────── */
  useEffect(() => {
    loadAnalytics();
    const iv = setInterval(loadAnalytics, 5000);
    return () => clearInterval(iv);
  }, []);

  const loadAnalytics = async () => {
    try {
      const analytics = await getAnalyticsSummary();
      const feed = await getAlerts();
      setSummary(analytics);
      setAlerts(feed || []);
    } catch (err) {
      console.error("[ANALYTICS ERROR]", err);
    }
  };

  /* ── KPI (unchanged data) ──────────────────────────── */
  const kpis = [
    { icon: <FaBroadcastTower />, title: "Active Cameras", value: "4" },
    { icon: <FaShieldAlt />, title: "Safety Score", value: `${summary?.safety_score || 98}%` },
    { icon: <FaExclamationTriangle />, title: "Critical Events", value: summary?.high || 0 },
    { icon: <FaServer />, title: "System Status", value: "ONLINE" },
  ];

  return (
    <PageLayout>
      <div className="an">

        {/* HEADER */}
        <motion.div className="an__hdr" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <div>
            <div className="an__eyebrow"><PDot /> ENTERPRISE ANALYTICS</div>
            <h1 className="an__h1">Operational<span className="an__accent"> Intelligence</span></h1>
            <p className="an__sub">Realtime AI operational intelligence</p>
          </div>
          <div className="an__status"><PDot /> AI OPERATIONAL</div>
        </motion.div>

        {/* KPI */}
        <div className="an__kpi">
          {kpis.map((item, i) => (
            <motion.div
              key={i}
              className="an__kpi-card"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.07 }}
              whileHover={{ y: -4 }}
            >
              <div className="an__kpi-icon">{item.icon}</div>
              <div>
                <div className="an__kpi-val">{item.value}</div>
                <div className="an__kpi-label">{item.title}</div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* MAIN GRID */}
        <div className="an__main">

          {/* INCIDENT FEED */}
          <motion.div className="an__panel" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <div className="an__panel-title"><FaBolt /> Live Incident Feed</div>
            <div className="an__feed">
              {alerts.length > 0
                ? alerts.slice(0, 8).map((item, i) => (
                  <div key={i} className={`an__incident ${incCls(item.risk_level)}`}>
                    <div className="an__incident-left">
                      <div className="an__incident-dot" />
                      <div>
                        <div className="an__incident-type">{item.event_type}</div>
                        <div className="an__incident-desc">{item.description}</div>
                      </div>
                    </div>
                    <div className="an__incident-risk">{item.risk_level}</div>
                  </div>
                ))
                : <div className="an__empty">No incidents detected</div>
              }
            </div>
          </motion.div>

          {/* TELEMETRY */}
          <motion.div className="an__panel" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.08 }}>
            <div className="an__panel-title"><FaWaveSquare /> AI Telemetry</div>
            <div className="an__telem-grid">
              {[
                { icon: <FaCheckCircle />, label: "WebSocket", val: "CONNECTED" },
                { icon: <FaChartLine />, label: "Processing", val: "LIVE" },
                { icon: <FaServer />, label: "AI Engine", val: "ACTIVE" },
                { icon: <FaBroadcastTower />, label: "Monitoring", val: "STABLE" },
              ].map((t, i) => (
                <div key={i} className="an__telem-cell">
                  {t.icon}
                  <div className="an__telem-label">{t.label}</div>
                  <div className="an__telem-val">{t.val}</div>
                </div>
              ))}
            </div>
          </motion.div>

        </div>
      </div>
    </PageLayout>
  );
}