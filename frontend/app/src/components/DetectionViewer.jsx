import "./detectionViewer.css";

const MAX_DETECTIONS = 8;

function DetectionViewer({ detections = [] }) {
  if (!detections.length) {
    return (
      <div className="detection-viewer">
        <div className="detection-empty">📡 No live detections yet</div>
      </div>
    );
  }

  return (
    <div className="detection-viewer" role="list">
      {detections.slice(0, MAX_DETECTIONS).map((det, i) => {
        const label = det.label || det.class_name || det.className || "Unknown Detection";
        const confidence = Number(det.confidence ?? det.conf ?? 0);
        const risk = (det.risk || "low").toLowerCase();

        return (
          <div
            key={det.id ?? `${label}-${confidence}`}
            className={`detection-card ${risk}`}
            role="listitem"
          >
            <div className="detection-top">
              <span className="detection-label">{label}</span>
              <span className={`detection-risk ${risk}`}>{risk}</span>
            </div>
            <div className="detection-meta">
              <span>{(confidence * 100).toFixed(1)}%</span>
              <div className="detection-bar">
                <div
                  className="detection-bar-fill"
                  style={{ width: `${(confidence * 100).toFixed(1)}%` }}
                />
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default DetectionViewer;