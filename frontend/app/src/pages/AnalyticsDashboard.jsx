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
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  /* ── load ───────────────────────────────────────────── */
  useEffect(() => {
    loadAnalytics();
    const iv = setInterval(loadAnalytics, 15000);
    return () => clearInterval(iv);
  }, []);

  const loadAnalytics = async () => {
    try {
      const analytics = await getAnalyticsSummary();
      const feed = await getAlerts();
      setSummary(analytics);
      setAlerts(feed || []);
      setError(null);
    } catch (err) {
      console.error("[ANALYTICS ERROR]", err);
      setError("Unable to load analytics");
    } finally {
      setLoading(false);
    }
  };

  /* ── KPI (now sourced from /activity/analytics/summary) ── */
  const kpis = [
    { icon: <FaShieldAlt />, title: "Safety Score", value: `${summary?.safety_score ?? "--"}%` },
    { icon: <FaExclamationTriangle />, title: "Total Alerts", value: summary?.total ?? "--" },
    { icon: <FaExclamationTriangle />, title: "High Risk Alerts", value: summary?.high ?? "--" },
    { icon: <FaBroadcastTower />, title: "Active Connections", value: summary?.active_connections ?? "--" },
    { icon: <FaChartLine />, title: "Events Sent", value: summary?.events_sent ?? "--" },
    { icon: <FaServer />, title: "System Status", value: summary?.system_status?.toUpperCase() ?? "UNKNOWN" },
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
          <div className="an__status-group">
            <div className="an__status"><PDot /> AI OPERATIONAL</div>
            {summary?.last_updated && (
              <div className="an__updated">
                Last Updated: {new Date(summary.last_updated).toLocaleTimeString()}
              </div>
            )}
          </div>
        </motion.div>

        {/* LOADING / ERROR */}
        {loading && (
          <div className="an__loading">Loading Analytics...</div>
        )}
        {!loading && error && (
          <div className="an__error">{error}</div>
        )}

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

        {/* RISK DISTRIBUTION */}
        {summary?.total === 0 ? (
          <div className="an__empty-state">
            No safety events detected yet. System operating normally.
          </div>
        ) : (
          <div className="an__risk-dist">
            <div className="an__risk-card an__risk-card--high">
              <div className="an__risk-val">{summary?.high ?? "--"}</div>
              <div className="an__risk-label">High</div>
            </div>
            <div className="an__risk-card an__risk-card--medium">
              <div className="an__risk-val">{summary?.medium ?? "--"}</div>
              <div className="an__risk-label">Medium</div>
            </div>
            <div className="an__risk-card an__risk-card--low">
              <div className="an__risk-val">{summary?.low ?? "--"}</div>
              <div className="an__risk-label">Low</div>
            </div>
          </div>
        )}

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
                        <div className="an__incident-meta">
                          {item.camera_id !== undefined && item.camera_id !== null && (
                            <span>{item.camera_id}</span>
                          )}
                          {item.timestamp && (
                            <span> · {new Date(item.timestamp).toLocaleTimeString()}</span>
                          )}
                        </div>
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
          {/*
            TODO: values below are still static placeholders.
            No confirmed field exists in /activity/analytics/summary
            for websocket/engine/pipeline status. If a
            /activity/analytics/health endpoint exists, wire it
            here once its response shape is confirmed (add
            getAnalyticsHealth() to services/api.js first).
          */}
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