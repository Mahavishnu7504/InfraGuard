import "./detectionViewer.css";

function DetectionViewer({ detections = [] }) {
  if (!detections.length) {
    return (
      <div className="detection-viewer">
        <div className="detection-empty">📡 No live detections yet</div>
      </div>
    );
  }

  return (
    <div className="detection-viewer">
      {detections.slice(0, 8).map((det, i) => {
        const label = det.label || det.class_name || det.className || "Unknown Detection";
        const confidence = Number(det.confidence ?? det.conf ?? 0);
        const risk = (det.risk || "low").toLowerCase();

        return (
          <div key={det.id || i} className={`detection-card ${risk}`}>
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