import { useEffect, useRef, useState } from "react";

const DEFAULT_DATA = {
  detections: [],
  risk: "LOW",
  camera_id: 1,
  running: false,
};

const getWSUrl = (cameraId) => {
  const isLocal =
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1";

  const base = isLocal
    ? "ws://127.0.0.1:8000"
    : "wss://infraguard-swyt.onrender.com";

  return `${base}/safety/ws/${cameraId}`;
};

export default function useWebSocket(
  cameraId = 1,
  enabled = true
) {
  const [data, setData] = useState(DEFAULT_DATA);
  const [alerts, setAlerts] = useState([]);
  const [connected, setConnected] = useState(false);

  const wsRef = useRef(null);
  const retryRef = useRef(null);
  // FIX (#2): mirrors the `enabled` prop into a ref so the onclose
  // handler (captured at connect time) can check the *current* value
  // rather than the value from when it was created — otherwise a
  // stale closure keeps reconnecting forever even after `enabled`
  // flips to false.
  const enabledRef = useRef(enabled);

  useEffect(() => {
    enabledRef.current = enabled;

    if (!enabled) {
      // FIX (#5): explicitly tear down any pending retry and close
      // the live socket the moment this hook is disabled (e.g. when
      // the camera is stopped), instead of leaving it open.
      clearTimeout(retryRef.current);
      wsRef.current?.close();
      wsRef.current = null;
      setConnected(false);
      return;
    }

    let mounted = true;

    const connect = () => {
      // Guard against connecting if we were disabled while a retry
      // was already queued.
      if (!enabledRef.current) return;

      const ws = new WebSocket(getWSUrl(cameraId));
      wsRef.current = ws;

      ws.onopen = () => {
        if (mounted) setConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);

          setData({
            detections: msg.detections || [],
            risk: msg.risk || "LOW",
            running: msg.running || false,
            camera_id: msg.camera_id || cameraId,
          });

          const detections = msg.detections || [];

          const liveAlerts = detections.map((d, i) => ({
            id: i,
            type: d.class_name || "Violation",
            risk: d.risk || msg.risk || "MEDIUM",
          }));

          setAlerts(liveAlerts);

        } catch (err) {
          console.error("WS Parse Error:", err);
        }
      };

      ws.onclose = () => {
        if (!mounted) return;

        setConnected(false);

        // FIX (#2): only schedule a reconnect if still enabled —
        // previously this always retried, so calling stop on a
        // camera never actually stopped the reconnect loop.
        if (enabledRef.current) {
          retryRef.current = setTimeout(connect, 2000);
        }
      };

      ws.onerror = () => ws.close();
    };

    connect();

    return () => {
      mounted = false;
      clearTimeout(retryRef.current);
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [cameraId, enabled]);

  return { data, alerts, connected };
}