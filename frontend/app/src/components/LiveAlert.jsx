import { FaCheckCircle } from "react-icons/fa";
import "./liveAlert.css";

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

  const limit = compact ? 4 : 12;

  return (
    <div className={`alert-container${compact ? " compact" : ""}`}>
      {alerts.slice(0, limit).map((alert, i) => {
        const risk = (alert.risk || "low").toLowerCase();
        const riskClass = risk === "high" ? "high" : risk === "medium" ? "medium" : "low";
        const title = alert.type || alert.event_type || alert.title || "AI Safety Alert";
        const time = alert.time || alert.timestamp || new Date().toLocaleTimeString();
        const desc = alert.message || alert.description || `${title} detected in monitoring pipeline`;

        return (
          <div key={alert.id || i} className={`alert-card ${riskClass}`}>
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