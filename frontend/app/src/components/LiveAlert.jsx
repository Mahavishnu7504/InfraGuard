import { FaCheckCircle } from "react-icons/fa";
import "./liveAlert.css";

const COMPACT_LIMIT = 4;
const DEFAULT_LIMIT = 12;

function formatTime(raw) {
  if (!raw) return new Date().toLocaleTimeString();
  const parsed = new Date(raw);
  return isNaN(parsed.getTime())
    ? raw
    : parsed.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function LiveAlert({ alerts = [], compact = false }) {
  if (!alerts.length) {
    return (
      <div className="alert-container">
        <div className="no-alerts">
          <FaCheckCircle /> No active alerts right now
        </div>
      </div>
    );
  }

  const limit = compact ? COMPACT_LIMIT : DEFAULT_LIMIT;

  return (
    <div className={`alert-container${compact ? " compact" : ""}`}>
      {alerts.slice(0, limit).map((alert, i) => {
        const risk = (alert.risk || "low").toLowerCase();
        const riskClass = risk === "high" ? "high" : risk === "medium" ? "medium" : "low";
        const title = alert.type || alert.event_type || alert.title || "AI Safety Alert";
        const time = formatTime(alert.timestamp || alert.time);
        const desc = alert.message || alert.description || `${title} detected in monitoring pipeline`;

        return (
          <div key={alert.id ?? `${title}-${time}`} className={`alert-card ${riskClass}`}>
            <div className="alert-header">
              <span>{title}</span>
              <span className={`alert-risk-badge ${riskClass}`}>{riskClass}</span>
            </div>
            <div className="alert-body">{desc}</div>
            <div className="alert-time">{time}</div>
          </div>
        );
      })}
    </div>
  );
}

export default LiveAlert;