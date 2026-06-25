import { useEffect, useMemo, useRef, useState } from "react";
import {
  FaUpload, FaShieldAlt, FaSearch,
  FaExclamationTriangle, FaCheckCircle, FaChartBar,
} from "react-icons/fa";
import { motion, AnimatePresence } from "framer-motion";

import PageLayout from "../components/PageLayout";
import { API_BASE } from "../services/api";
import "./infraDetection.css";

/* ─── risk helpers ──────────────────────────────────────── */
const FC = { HIGH: "id__finding--high", MEDIUM: "id__finding--medium", LOW: "id__finding--low" };
const fc = (r) => FC[(r || "LOW").toUpperCase()] || FC.LOW;

/* ─── upload constraints ────────────────────────────────── */
const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/jpg", "image/webp"];
const MAX_FILE_SIZE_MB = 10;
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;

export default function InfraDetection() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [annotatedPreview, setAnnotatedPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  // Tracks the object URL currently in use so it can be revoked on
  // replacement/unmount without retriggering effects off `preview` itself.
  const previewUrlRef = useRef(null);

  /* ── revoke any outstanding object URL on unmount ──────── */
  useEffect(() => {
    return () => {
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current);
      }
    };
  }, []);

  /* ── findings (same logic) ─────────────────────────── */
  const findings = useMemo(() => {
    if (!result?.detections?.length) return [];
    return result.detections.map((det, idx) => ({
      id: idx,
      title: det.label || "Detected Object",
      severity: (det.risk || "low").toUpperCase(),
      confidence: det.confidence || 0,
    }));
  }, [result]);

  const stats = useMemo(() => ({
    high: findings.filter(x => x.severity === "HIGH").length,
    medium: findings.filter(x => x.severity === "MEDIUM").length,
    low: findings.filter(x => x.severity === "LOW").length,
  }), [findings]);

  /* ── file upload (validated + leak-safe) ───────────── */
  const handleUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!ALLOWED_TYPES.includes(file.type)) {
      setError("Unsupported file type. Please upload a JPG, PNG, or WEBP image.");
      e.target.value = "";
      return;
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      setError(`File too large. Maximum size is ${MAX_FILE_SIZE_MB}MB.`);
      e.target.value = "";
      return;
    }

    // Revoke the previous preview URL before creating a new one.
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
    }

    const objectUrl = URL.createObjectURL(file);
    previewUrlRef.current = objectUrl;

    setSelectedFile(file);
    setPreview(objectUrl);
    setAnnotatedPreview(null);
    setResult(null);
    setError("");
  };

  /* ── detection (guarded + better error surfacing) ──── */
  const runDetection = async () => {
    if (!selectedFile) { alert("Upload image first"); return; }
    if (loading) return; // prevent duplicate/rapid-fire requests

    setLoading(true); setError("");
    try {
      const fd = new FormData();
      fd.append("file", selectedFile);
      const res = await fetch(`${API_BASE}/safety/detect-full`, { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || `Request failed (${res.status})`);
      setResult(data);
      setAnnotatedPreview(`data:image/jpeg;base64,${data.image}`);
    } catch (err) {
      console.error("Detection request failed:", err);
      setError(err?.message || "Server waking up — try again in a few seconds.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageLayout>
      <div className="id">

        {/* ── HEADER ── */}
        <motion.div className="id__hdr" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <div className="id__eyebrow"><FaShieldAlt /> AI SAFETY ENGINE</div>
          <h1 className="id__h1">Safety<span className="id__accent"> Detection Center</span></h1>
          <p className="id__sub">
            Upload construction or industrial images — InfraGuard AI performs
            PPE compliance checks and safety risk analysis.
          </p>
        </motion.div>

        {/* ── MAIN GRID ── */}
        <div className="id__grid">

          {/* LEFT — upload panel */}
          <motion.div
            className="id__panel"
            initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
          >
            <div className="id__panel-title"><FaUpload /> Upload Center</div>

            <input type="file" id="id-upload" className="id__upload-hidden"
              accept="image/*" onChange={handleUpload} />

            <label htmlFor="id-upload" className="id__dropzone">
              <div className="id__dropzone-icon"><FaUpload /></div>
              <div>
                <div className="id__dropzone-name">
                  {selectedFile ? selectedFile.name : "Choose Inspection Image"}
                </div>
                <div className="id__dropzone-hint">JPG · PNG · JPEG</div>
              </div>
            </label>

            <div className="id__actions">
              <button className="id__btn id__btn--run" disabled={loading} onClick={runDetection}>
                <FaSearch /> {loading ? "Analyzing…" : "Run Detection"}
              </button>
            </div>

            {error && <div className="id__note">{error}</div>}

            <div className="id__mini-kpi">
              <div className="id__mkpi id__mkpi--h">
                <div className="id__mkpi-label">High Risk</div>
                <div className="id__mkpi-val">{stats.high}</div>
              </div>
              <div className="id__mkpi id__mkpi--m">
                <div className="id__mkpi-label">Medium Risk</div>
                <div className="id__mkpi-val">{stats.medium}</div>
              </div>
              <div className="id__mkpi id__mkpi--l">
                <div className="id__mkpi-label">Low Risk</div>
                <div className="id__mkpi-val">{stats.low}</div>
              </div>
            </div>
          </motion.div>

          {/* RIGHT — image viewport */}
          <motion.div
            className="id__panel"
            initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.08 }}
          >
            <div className="id__panel-title"><FaShieldAlt /> Live Inspection View</div>
            <div className="id__viewport">
              {preview ? (
                <img
                  src={annotatedPreview || preview}
                  alt="preview"
                  className={`id__feed-img${result?.risk === "HIGH" ? " id__feed-img--alert" : ""}`}
                />
              ) : (
                <div className="id__idle">
                  <FaShieldAlt className="id__idle-icon" />
                  <div className="id__idle-title">Awaiting Inspection Image</div>
                  <div className="id__idle-sub">Upload an image to begin AI analysis</div>
                </div>
              )}
            </div>
          </motion.div>
        </div>

        {/* ── ANALYZING STATE ── */}
        <AnimatePresence>
          {loading && (
            <motion.div
              className="id__panel"
              initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            >
              <div className="id__spin-wrap">
                <div className="id__spinner" />
                <div className="id__spin-title">AI Analysis In Progress</div>
                <div className="id__spin-sub">Running detection models and compliance validation</div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── RESULTS ── */}
        <AnimatePresence>
          {result && !loading && (
            <motion.div
              className="id__results"
              initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
            >
              {/* summary cards */}
              <div className="id__summary-grid">
                {[
                  { icon: <FaShieldAlt />, label: "Risk Level", val: result.risk || "LOW" },
                  { icon: <FaChartBar />, label: "Detections", val: result.detections?.length || 0 },
                  { icon: <FaCheckCircle />, label: "AI Status", val: "COMPLETE" },
                ].map((c, i) => (
                  <div key={i} className="id__sum-card">
                    <div className="id__sum-icon">{c.icon}</div>
                    <div>
                      <div className="id__sum-label">{c.label}</div>
                      <div className="id__sum-val">{c.val}</div>
                    </div>
                  </div>
                ))}
              </div>

              {/* ai summary + findings */}
              <div className="id__result-grid">
                <div className="id__panel">
                  <div className="id__panel-title"><FaShieldAlt /> AI Summary</div>
                  <div className="id__ai-box">
                    <div className="id__ai-box-label">Analysis Result</div>
                    <div className="id__ai-box-text">
                      Detection completed successfully. Review findings and compliance indicators.
                    </div>
                  </div>
                </div>

                <div className="id__panel">
                  <div className="id__panel-title"><FaExclamationTriangle /> Findings</div>
                  <div className="id__findings">
                    {findings.length > 0
                      ? findings.map(item => (
                        <div key={item.id} className={`id__finding ${fc(item.severity)}`}>
                          <div className="id__finding-top">
                            <div className="id__finding-icon"><FaExclamationTriangle /></div>
                            <div>
                              <div className="id__finding-title">{item.title}</div>
                              <div className="id__finding-badge">{item.severity}</div>
                            </div>
                          </div>
                          <div className="id__finding-conf">
                            Confidence: {(item.confidence * 100).toFixed(1)}%
                          </div>
                        </div>
                      ))
                      : <div className="id__ai-box"><div className="id__ai-box-text">No findings detected.</div></div>
                    }
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

      </div>
    </PageLayout>
  );
}