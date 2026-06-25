import { useEffect, useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

import {
  FaBroadcastTower, FaPlay, FaStop, FaShieldAlt,
  FaPlus, FaVideo, FaExclamationTriangle,
  FaServer, FaBolt, FaEye, FaTimes, FaCheck,
} from "react-icons/fa";
import { MdOutlineGridView, MdGrid3X3, MdGridOn } from "react-icons/md";

import PageLayout from "../components/PageLayout";
import { getCameraFeedUrl, startCamera, stopCamera, getCameraStatus } from "../services/api";
import useWebSocket from "../hooks/useWebSocket";
import "./liveCamera.css";

/* ─── risk config ─────────────────────────────────────── */
const RISK = {
  LOW: { cls: "r-low", hex: "#00ff9d", label: "LOW" },
  MEDIUM: { cls: "r-medium", hex: "#f59e0b", label: "MEDIUM" },
  HIGH: { cls: "r-high", hex: "#ef4444", label: "HIGH" },
  CRITICAL: { cls: "r-critical", hex: "#ff0055", label: "CRITICAL" },
};
const getRisk = (r) => RISK[(r || "LOW").toUpperCase()] ?? RISK.LOW;

/* ─── default cameras ─────────────────────────────────── */
const DEFAULT_CAMERAS = [
  { id: 0, name: "North Gate", location: "Main Entry — Zone A" },
  { id: 1, name: "South Perimeter", location: "Exit Point — Zone B" },
  { id: 2, name: "Warehouse Floor", location: "Industrial Zone — C" },
  { id: 3, name: "Crane Zone", location: "Restricted — Zone D" },
];
const MAX_CAMERAS = 8;

// How often each panel re-checks backend status to catch drift
// (backend crash, camera unplugged, etc.) that local state alone
// wouldn't notice.
const STATUS_POLL_MS = 5000;

/* ═══════════════════════════════════════════════════════
   SCANLINE OVERLAY
═══════════════════════════════════════════════════════ */
function ScanlineOverlay() {
  return <div className="scanline" aria-hidden="true" />;
}

/* ═══════════════════════════════════════════════════════
   HUD CORNER BRACKETS
═══════════════════════════════════════════════════════ */
function HudCorners({ active }) {
  return (
    <div className={`hud-corners${active ? " hud-corners--on" : ""}`} aria-hidden="true">
      <span className="hc tl" /><span className="hc tr" />
      <span className="hc bl" /><span className="hc br" />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════
   PULSE DOT
═══════════════════════════════════════════════════════ */
function PulseDot({ color = "#00ff9d" }) {
  return <span className="pdot" style={{ "--c": color }} aria-hidden="true" />;
}

/* ═══════════════════════════════════════════════════════
   CAMERA PANEL
═══════════════════════════════════════════════════════ */
function CameraPanel({ cam, onStart, onStop, onRemove }) {
  const [streamKey, setStreamKey] = useState(null);
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);
  const [uptime, setUptime] = useState(0);
  const timerRef = useRef(null);
  const retryCountRef = useRef(0);
  const retryTimeoutRef = useRef(null);
  const MAX_RETRIES = 5;

  // FIX (#4, #12): each panel owns its own websocket scoped to its
  // own camera id, instead of every panel sharing one socket hardcoded
  // to camera 0. The socket is only enabled while this panel believes
  // its camera is running, so stopping the camera also closes the
  // socket (see fix in useWebSocket.js for the close-on-disable side).
  const { data: live, connected: wsConnected } = useWebSocket(cam.id, running);

  const r = getRisk(live?.risk || "LOW");
  const fps = live?.telemetry?.fps || 0;
  const dets = live?.detections?.length || 0;

  // FIX (#3): periodically reconcile local `running` state against
  // the backend's actual status, so a backend crash or camera drop
  // can't leave the UI showing "LIVE" indefinitely.
  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      try {
        const status = await getCameraStatus(cam.id);
        if (cancelled) return;

        setRunning((prev) => {
          if (prev && !status.running) {
            setStreamKey(null);
            setError(true);
          }
          return status.running;
        });
      } catch {
        // A single failed status check shouldn't flip state — could
        // just be a transient network blip.
      }
    };

    const interval = setInterval(poll, STATUS_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [cam.id]);

  useEffect(() => {
    if (running) {
      timerRef.current = setInterval(() => setUptime(u => u + 1), 1000);
    } else {
      clearInterval(timerRef.current);
      setUptime(0);
    }
    return () => clearInterval(timerRef.current);
  }, [running]);

  const fmt = (s) => {
    const h = String(Math.floor(s / 3600)).padStart(2, "0");
    const m = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
    const sc = String(s % 60).padStart(2, "0");
    return `${h}:${m}:${sc}`;
  };

  const handleStart = async () => {
    if (running || loading) return;
    setError(false); setLoading(true);
    const ok = await onStart(cam.id);
    setLoading(false);
    if (ok) { retryCountRef.current = 0; setRunning(true); setStreamKey(Date.now()); }
    else setError(true);
  };

  const handleStop = async () => {
    if (!running || loading) return;
    clearTimeout(retryTimeoutRef.current);
    retryCountRef.current = 0;
    await onStop(cam.id);
    // FIX (#5): setting running=false disables this panel's
    // useWebSocket hook, which closes its socket immediately rather
    // than leaving it connected after a stop.
    setRunning(false);
    setStreamKey(null);
  };

  // A single dropped MJPEG connection (network blip, slow frame, proxy
  // timeout) should not be treated as a fatal stream failure. Retry with
  // backoff by reloading the <img> via a fresh streamKey before giving up.
  const handleStreamError = () => {
    if (retryCountRef.current >= MAX_RETRIES) {
      setError(true);
      setRunning(false);
      return;
    }
    retryCountRef.current += 1;
    const delay = Math.min(500 * 2 ** (retryCountRef.current - 1), 4000);
    clearTimeout(retryTimeoutRef.current);
    retryTimeoutRef.current = setTimeout(() => {
      setStreamKey(Date.now());
    }, delay);
  };

  useEffect(() => () => clearTimeout(retryTimeoutRef.current), []);

  return (
    <motion.div
      className={`cp${running ? " cp--live" : ""}${error ? " cp--err" : ""}`}
      initial={{ opacity: 0, y: 28 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.94 }}
      transition={{ duration: 0.36, ease: [0.22, 1, 0.36, 1] }}
      layout
    >
      <HudCorners active={running} />

      {/* HEAD */}
      <div className="cp__head">
        <div>
          <div className="cp__cam-id">CAM·{String(cam.id).padStart(2, "0")}</div>
          <div className="cp__name-row">
            <h3 className="cp__name">{cam.name}</h3>
            {running && <PulseDot />}
          </div>
          <div className="cp__loc">{cam.location}</div>
        </div>
        <div className="cp__head-r">
          <div className={`cp__badge${running ? " cp__badge--live" : " cp__badge--off"}`}>
            {running ? <><PulseDot />&nbsp;LIVE</> : "OFFLINE"}
          </div>
          <button className="cp__del" title="Remove" onClick={() => { if (window.confirm(`Remove ${cam.name}?`)) { handleStop(); onRemove(cam.id); } }}>
            <FaTimes />
          </button>
        </div>
      </div>

      {/* VIEWPORT */}
      <div className="cp__vp">
        {running && streamKey ? (
          <>
            <img
              key={streamKey}
              src={`${getCameraFeedUrl(cam.id)}&t=${streamKey}`}
              className="cp__feed"
              alt={`Cam ${cam.id}`}
              onError={handleStreamError}
            />
            <ScanlineOverlay />
            <div className="cp__risk-wash" style={{ "--rh": r.hex }} />
            <div className="cp__ai-chip"><FaShieldAlt /> AI ACTIVE</div>
            <div className={`cp__risk-badge ${r.cls}`}>{r.label}</div>
            <div className="cp__hud-bar">
              <span><FaBolt /> {fps} FPS</span>
              <span><FaEye /> {dets} OBJ</span>
              <span>⏱ {fmt(uptime)}</span>
            </div>
          </>
        ) : loading ? (
          <div className="cp__idle">
            <div className="cp__spin" />
            <span>Connecting…</span>
          </div>
        ) : (
          <div className="cp__idle">
            {error
              ? <><FaExclamationTriangle className="cp__idle-icon cp__idle-icon--err" /><span>Stream Unavailable</span><small>Camera {cam.id} not responding</small></>
              : <><FaBroadcastTower className="cp__idle-icon" /><span>Standby</span><small>Press START to connect</small></>
            }
          </div>
        )}
      </div>

      {/* CONTROLS */}
      <div className="cp__ctrl">
        <button className="cp__btn cp__btn--go" onClick={handleStart} disabled={running || loading}>
          <FaPlay /> START
        </button>
        <button className="cp__btn cp__btn--halt" onClick={handleStop} disabled={!running}>
          <FaStop /> STOP
        </button>
      </div>

      {/* TELEMETRY */}
      <div className="cp__telem">
        <div className="cp__t-cell">
          <span className="cp__t-k">FPS</span>
          <span className="cp__t-v">{fps}</span>
        </div>
        <div className={`cp__t-cell ${r.cls}`}>
          <span className="cp__t-k">RISK</span>
          <span className="cp__t-v">{r.label}</span>
        </div>
        <div className="cp__t-cell">
          <span className="cp__t-k">OBJ</span>
          <span className="cp__t-v">{dets}</span>
        </div>
        <div className="cp__t-cell">
          <span className="cp__t-k">UPTIME</span>
          <span className="cp__t-v">{fmt(uptime)}</span>
        </div>
      </div>
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════
   ADD CAMERA MODAL
═══════════════════════════════════════════════════════ */
function AddCameraModal({ existing, onAdd, onClose }) {
  const [camId, setCamId] = useState("");
  const [name, setName] = useState("");
  const [loc, setLoc] = useState("");
  const [err, setErr] = useState("");

  const submit = () => {
    const id = parseInt(camId, 10);
    if (isNaN(id) || id < 0 || id > 99) { setErr("Camera ID must be 0–99"); return; }
    if (existing.includes(id)) { setErr(`ID ${id} already registered`); return; }
    if (name && !name.trim()) { setErr("Display name can't be just spaces"); return; }
    if (loc && !loc.trim()) { setErr("Location can't be just spaces"); return; }
    onAdd({ id, name: name.trim() || `Camera ${String(id).padStart(2, "0")}`, location: loc.trim() || `Camera ${id}` });
    onClose();
  };

  return (
    <motion.div className="mbg" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose}>
      <motion.div
        className="mbox"
        initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }}
        transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
        onClick={e => e.stopPropagation()}
      >
        <div className="mbox__hd">
          <FaVideo className="mbox__ico" />
          <h2>Register Camera</h2>
          <button className="mbox__x" onClick={onClose}><FaTimes /></button>
        </div>
        <div className="mbox__bd">
          <label>Camera ID (0–99)</label>
          <input type="number" min={0} max={99} placeholder="e.g. 1" value={camId} onChange={e => { setCamId(e.target.value); setErr(""); }} />
          <label>Display Name</label>
          <input type="text" placeholder="e.g. South Gate" value={name} onChange={e => setName(e.target.value)} />
          <label>Location</label>
          <input type="text" placeholder="e.g. Zone B — Entry" value={loc} onChange={e => setLoc(e.target.value)} />
          {err && <p className="mbox__err">{err}</p>}
        </div>
        <div className="mbox__ft">
          <button className="mbox__cancel" onClick={onClose}>Cancel</button>
          <button className="mbox__add" onClick={submit}><FaPlus /> Register</button>
        </div>
      </motion.div>
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════
   MAIN PAGE
═══════════════════════════════════════════════════════ */
export default function LiveCamera() {
  const [cameras, setCameras] = useState(DEFAULT_CAMERAS);
  const [toasts, setToasts] = useState([]);
  const [modal, setModal] = useState(false);
  const [cols, setCols] = useState("2");
  const [now, setNow] = useState(new Date());

  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const toast = useCallback((type, msg) => {
    const id = Date.now() + Math.random();
    setToasts(p => [...p, { id, type, msg }]);
    setTimeout(() => setToasts(p => p.filter(x => x.id !== id)), 3500);
  }, []);

  const handleStart = useCallback(async (id) => {
    try { await startCamera(id); toast("ok", `Camera ${id} active`); return true; }
    catch { toast("err", `Camera ${id} failed`); return false; }
  }, [toast]);

  const handleStop = useCallback(async (id) => {
    try { await stopCamera(id); toast("warn", `Camera ${id} stopped`); }
    catch { toast("err", `Stop failed`); }
  }, [toast]);

  const handleRemove = useCallback((id) => { setCameras(p => p.filter(c => c.id !== id)); toast("warn", `Camera ${id} removed`); }, [toast]);
  const handleAdd = useCallback((cam) => { setCameras(p => [...p, cam]); toast("ok", `Camera ${cam.id} registered`); }, [toast]);

  return (
    <PageLayout>
      <div className="lp">

        {/* HEADER */}
        <header className="lp__hdr">
          <div className="lp__hdr-l">
            <div className="lp__eyebrow"><PulseDot /> INFRAGUARD — LIVE SURVEILLANCE</div>
            <h1 className="lp__h1">Command<span className="lp__accent"> Center</span></h1>
            <p className="lp__sub">Enterprise AI · Multi-Camera · Real-time Intelligence</p>
          </div>
          <div className="lp__hdr-r">
            <div className="lp__clock">
              <span className="lp__clock-t">{now.toLocaleTimeString("en-GB")}</span>
              <span className="lp__clock-d">{now.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" })}</span>
            </div>
          </div>
        </header>

        {/* TOOLBAR */}
        <div className="lp__bar">
          <div className="lp__chips">
            <div className="chip"><FaVideo /><strong>{cameras.length}</strong><small>Cameras</small></div>
            <div className="chip"><FaShieldAlt /><strong>ON</strong><small>AI Engine</small></div>
            <div className="chip chip--dim"><FaServer /><strong>v5.3</strong><small>Platform</small></div>
          </div>
          <div className="lp__tools">
            <div className="lp__gcols">
              {[{ v: "1", i: <MdGridOn /> }, { v: "2", i: <MdOutlineGridView /> }, { v: "3", i: <MdGrid3X3 /> }].map(({ v, i }) => (
                <button key={v} className={`lp__gcol-btn${cols === v ? " lp__gcol-btn--on" : ""}`} onClick={() => setCols(v)} title={`${v} columns`}>{i}</button>
              ))}
            </div>
            {cameras.length < MAX_CAMERAS && (
              <button className="lp__add" onClick={() => setModal(true)}>
                <FaPlus /> Add Camera
              </button>
            )}
          </div>
        </div>

        {/* GRID */}
        <div className="lp__grid" style={{ "--cols": cols }}>
          <AnimatePresence mode="popLayout">
            {cameras.map(cam => (
              <CameraPanel key={cam.id} cam={cam}
                onStart={handleStart} onStop={handleStop}
                onRemove={handleRemove}
              />
            ))}
            {cameras.length < MAX_CAMERAS && (
              <motion.div key="add-slot" className="cp cp--slot"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                onClick={() => setModal(true)}
              >
                <FaPlus className="slot-icon" />
                <span>Register Camera</span>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* MODAL */}
        <AnimatePresence>
          {modal && <AddCameraModal existing={cameras.map(c => c.id)} onAdd={handleAdd} onClose={() => setModal(false)} />}
        </AnimatePresence>

        {/* TOASTS */}
        <div className="lp__toasts">
          <AnimatePresence>
            {toasts.map(n => (
              <motion.div key={n.id} className={`lp__toast lp__toast--${n.type}`}
                initial={{ opacity: 0, x: 56 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 56 }}
                transition={{ duration: 0.22 }}
              >
                <span className="lp__toast-ico">
                  {n.type === "ok" ? <FaCheck /> : n.type === "warn" ? <FaBolt /> : <FaExclamationTriangle />}
                </span>
                {n.msg}
              </motion.div>
            ))}
          </AnimatePresence>
        </div>

      </div>
    </PageLayout>
  );
}