import {
  useRef,
  useState,
  useMemo,
  useEffect,
  useCallback,
} from "react";

import {
  FaMagic,
  FaRedo,
  FaSpinner,
  FaCheckCircle,
  FaCamera,
  FaShieldAlt,
  FaExclamationTriangle,
  FaBrain,
  FaChartLine,
  FaTimes,
  FaPlus,
} from "react-icons/fa";

import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router-dom";

import PageLayout from "../../components/PageLayout";
import PageCard from "../../components/PageCard";
import { analyzeQuality } from "../../services/api";

import "../quality.css";

/* ── Workflow progress shared component ──────────────────────────── */
function WorkflowProgress({ current = 1 }) {
  const steps = [
    { label: "Overview" },
    { label: "Upload" },
    { label: "Practices" },
    { label: "Report" },
  ];
  return (
    <div className="workflow-progress">
      {steps.map((s, i) => {
        const isDone = i < current;
        const isActive = i === current;
        return (
          <div key={i} className={`wp-step ${isActive ? "active" : ""} ${isDone ? "done" : ""}`}>
            {i > 0 && <div className={`wp-connector ${isDone ? "done" : ""}`} />}
            <div className={`wp-bubble ${isActive ? "active" : ""} ${isDone ? "done" : ""}`}>
              {isDone ? <FaCheckCircle style={{ fontSize: "0.65rem" }} /> : `0${i + 1}`}
            </div>
            <span className="wp-label">{s.label}</span>
          </div>
        );
      })}
    </div>
  );
}

/* ── Processing stage messages ─────────────────────────────────── */
const STAGES = [
  "Initializing Enterprise Inspection Engine",
  "Uploading Visual Evidence",
  "Running AI Structural Analysis",
  "Executing Operational Risk Assessment",
  "Generating Intelligence Report",
  "Finalizing Inspection Analytics",
];

export default function UploadImages() {
  const navigate = useNavigate();
  const inputRef = useRef(null);
  const dropRef = useRef(null);

  /* ── STATE ── */
  const [files, setFiles] = useState([]);     // { file, preview, id }[]
  const [activeIdx, setActiveIdx] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [processingStep, setProcessingStep] = useState("");
  const [stageProgress, setStageProgress] = useState(0);
  const [dragOver, setDragOver] = useState(false);

  /* ── PROCESSING STAGE ANIMATION ── */
  useEffect(() => {
    if (!loading) return;
    let index = 0;
    setProcessingStep(STAGES[0]);
    setStageProgress(10);

    const interval = setInterval(() => {
      index++;
      if (index < STAGES.length) {
        setProcessingStep(STAGES[index]);
        setStageProgress(Math.round((index / (STAGES.length - 1)) * 100));
      }
    }, 1100);

    return () => clearInterval(interval);
  }, [loading]);

  /* ── FILE HELPERS ── */
  const addFiles = useCallback((selected) => {
    const valid = Array.from(selected).filter(f => f.type.startsWith("image/"));
    if (!valid.length) return;
    const entries = valid.map(f => ({
      id: Math.random().toString(36).slice(2),
      file: f,
      preview: URL.createObjectURL(f),
    }));
    setFiles(prev => {
      const next = [...prev, ...entries];
      setActiveIdx(next.length - 1);
      return next;
    });
    setError("");
    setResult(null);
  }, []);

  const removeFile = (id) => {
    setFiles(prev => {
      const next = prev.filter(f => f.id !== id);
      setActiveIdx(i => Math.min(i, Math.max(0, next.length - 1)));
      return next;
    });
  };

  /* ── INPUT ── */
  const handleInput = (e) => addFiles(e.target.files);

  /* ── DRAG & DROP ── */
  const handleDragOver = (e) => { e.preventDefault(); setDragOver(true); };
  const handleDragLeave = () => setDragOver(false);
  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    addFiles(e.dataTransfer.files);
  };

  /* ── ANALYZE ──
     FIX: All images are appended under the same field name "files".
     The backend expects: files: List[UploadFile]
     So every image must use fd.append("files", image.file).
     The old split of "file" (primary) + "files" (rest) caused a 422.
  ── */
  const analyze = async () => {
    if (!files.length) {
      setError("Upload at least one inspection image before analysis.");
      return;
    }

    try {
      setLoading(true);
      setError("");

      const fd = new FormData();

      // Send ALL images under the unified "files" field
      files.forEach((f) => {
        fd.append("files", f.file);
      });

      const data = await analyzeQuality(fd);

      if (!data?.success) {
        throw new Error(data?.error || "Enterprise inspection failed.");
      }

      // The backend now returns { success, images: [...], ... }
      // Persist the full response so the report page has all image results.
      localStorage.setItem("qualityData", JSON.stringify(data));
      // Use the active image's preview as the primary preview.
      localStorage.setItem(
        "qualityImagePreview",
        files[activeIdx]?.preview || files[0]?.preview || ""
      );

      setResult(data);
    } catch (err) {
      console.error(err);
      setError(err?.message || "Enterprise inspection service unavailable.");
    } finally {
      setLoading(false);
      setStageProgress(0);
    }
  };

  /* ── RESET ── */
  const reset = () => {
    files.forEach(f => URL.revokeObjectURL(f.preview));
    setFiles([]);
    setActiveIdx(0);
    setResult(null);
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  };

  /* ── DERIVED SUMMARY FROM FIRST PROCESSED IMAGE ──
     FIX: The backend wraps per-image results inside result.images[].
     The old code read result.compliance_score etc. directly from the
     top-level response, which no longer exists in the multi-image format.
     We derive summary display values from images[0] as the primary result.
  ── */
  const primaryImage = result?.images?.[0] || null;

  const grade = useMemo(() => {
    const score = primaryImage?.compliance_score || 0;
    if (score >= 95) return "A+";
    if (score >= 90) return "A";
    if (score >= 80) return "B";
    if (score >= 70) return "C";
    return "D";
  }, [primaryImage]);

  /* ── ACTIVE PREVIEW ── */
  const activePreview = files[activeIdx]?.preview || "";

  /* ── UI ── */
  return (
    <PageLayout
      badge="Enterprise AI Inspection"
      title="Inspection Intelligence Center"
      subtitle="Realtime AI-powered construction quality analysis and operational compliance intelligence"
    >
      <div className="minimal-page">

        {/* ── PROGRESS ── */}
        <WorkflowProgress current={1} />

        {/* ── BACK ── */}
        <div style={{ marginBottom: 16 }}>
          <button
            className="secondary-btn"
            onClick={() => navigate("/quality")}
          >
            ← Back
          </button>
        </div>

        {/* ── HERO ── */}
        <motion.div
          className="enterprise-upload-hero"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="hero-left">
            <h1>Upload Inspection Images</h1>
            <p>
              Upload one or more site images for AI-powered quality inspection.
            </p>
          </div>
        </motion.div>

        {/* ── UPLOAD PANEL ── */}
        <motion.div
          className="minimal-upload-panel"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.08 }}
        >
          {/* Drop zone */}
          <div
            ref={dropRef}
            className={`minimal-upload-zone ${dragOver ? "drag-over" : ""}`}
            onClick={() => !files.length && inputRef.current?.click()}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            {files.length > 0 ? (
              activePreview
                ? <img src={activePreview} className="minimal-preview" alt="inspection preview" />
                : null
            ) : (
              <div className="upload-placeholder">
                <FaCamera />
                <h3>Upload Inspection Evidence</h3>
                <p>Drag & drop or click to select · JPG, PNG, WEBP · multiple files supported</p>
              </div>
            )}
            <input
              hidden
              ref={inputRef}
              type="file"
              accept="image/*"
              multiple
              onChange={handleInput}
            />
          </div>

          {/* Gallery strip (multi-image) */}
          <AnimatePresence>
            {files.length > 0 && (
              <motion.div
                className="image-gallery"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                style={{ marginBottom: 16 }}
              >
                {files.map((f, i) => (
                  <div
                    key={f.id}
                    className={`gallery-thumb ${i === activeIdx ? "active-thumb" : ""}`}
                    onClick={() => setActiveIdx(i)}
                  >
                    <img src={f.preview} alt={`thumb-${i}`} />
                    <button
                      className="gallery-thumb-remove"
                      onClick={e => { e.stopPropagation(); removeFile(f.id); }}
                    >
                      <FaTimes />
                    </button>
                  </div>
                ))}
                {/* Add more button */}
                <div
                  className="gallery-thumb"
                  style={{
                    display: "flex", alignItems: "center", justifyContent: "center",
                    cursor: "pointer", border: "1.5px dashed var(--border-hi)",
                    background: "rgba(0,200,255,0.03)",
                  }}
                  onClick={() => inputRef.current?.click()}
                >
                  <FaPlus style={{ color: "var(--b)", fontSize: "1rem" }} />
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Processing stage */}
          <AnimatePresence>
            {loading && (
              <motion.div
                className="enterprise-processing"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <FaSpinner className="spin" />
                <div style={{ flex: 1 }}>
                  <h4>AI Inspection Processing</h4>
                  <p>{processingStep}</p>
                  <div className="upload-progress" style={{ marginTop: 10 }}>
                    <div
                      className="upload-progress-fill"
                      style={{ width: `${stageProgress}%` }}
                    />
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Actions */}
          <div className="hero-actions">
            <button
              className="primary-btn"
              onClick={analyze}
              disabled={loading}
            >
              {loading ? <FaSpinner className="spin" /> : <FaMagic />}
              {loading ? "Analyzing..." : "Run AI Analysis"}
            </button>

            <button className="secondary-btn" onClick={reset} disabled={loading}>
              <FaRedo /> Reset
            </button>
          </div>

          {/* Error */}
          {error && (
            <motion.div
              className="error-box"
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <FaExclamationTriangle />
              {error}
            </motion.div>
          )}
        </motion.div>

        {/* ── RESULT ──
            FIX: All display values now read from result.images[0] (primaryImage)
            instead of the top-level result object, which no longer carries
            per-image fields in the multi-image response format.
        ── */}
        <AnimatePresence>
          {result && primaryImage && (
            <motion.div
              className="minimal-result"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
            >
              {/* Score cards */}
              <div className="executive-grid">
                {[
                  {
                    icon: <FaShieldAlt />,
                    label: "Compliance Score",
                    value: `${primaryImage.compliance_score}/100`,
                  },
                  {
                    icon: <FaCheckCircle />,
                    label: "Inspection Grade",
                    value: grade,
                  },
                  {
                    icon: <FaExclamationTriangle />,
                    label: "Operational Risk",
                    value: primaryImage.overall_risk,
                  },
                  {
                    icon: <FaBrain />,
                    label: "AI Findings",
                    value: primaryImage.report?.length || 0,
                  },
                ].map((c, i) => (
                  <PageCard key={i} className="executive-card">
                    {c.icon}
                    <div>
                      <h3>{c.value}</h3>
                      <span>{c.label}</span>
                    </div>
                  </PageCard>
                ))}
              </div>

              {/* Multi-image notice */}
              {result.total_images_processed > 1 && (
                <PageCard className="executive-summary-card">
                  <div className="report-section-header">
                    <FaChartLine /> Multi-Image Inspection
                  </div>
                  <p className="executive-summary-text">
                    {result.total_images_processed} images processed successfully.
                    Showing summary for image 1. Full per-image breakdown available in the report.
                    {result.errors?.length > 0 && (
                      <> {result.errors.length} image(s) could not be processed.</>
                    )}
                  </p>
                </PageCard>
              )}

              {/* Summary */}
              <PageCard className="executive-summary-card">
                <div className="report-section-header">
                  <FaBrain /> Executive Intelligence Summary
                </div>
                <p className="executive-summary-text">
                  {primaryImage.executive_summary || "Enterprise inspection completed successfully."}
                </p>
              </PageCard>

              {/* Navigation */}
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  marginTop: 20,
                }}
              >
                <button
                  className="secondary-btn"
                  onClick={() => navigate("/quality/report")}
                >
                  Skip to Report
                </button>

                <button
                  className="primary-btn"
                  onClick={() => navigate("/quality/best-practices")}
                >
                  Next →
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

      </div>
    </PageLayout>
  );
}