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
  FaCircle,
  FaFileImage,
  FaIdBadge,
  FaLayerGroup,
  FaFileAlt,
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

/* ── Tunable constants ────────────────────────────────────────────
   Centralized here so behavior (timing, limits, truncation) can be
   adjusted without hunting through JSX/logic for inline literals.
── */
const STAGE_INTERVAL_MS = 1100;
const MAX_FILENAME_LENGTH = 22;
const TRUNCATED_FILENAME_LENGTH = 19;
const MAX_IMAGES = 10;
const MAX_FILE_SIZE_MB = 10;

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

  // Enterprise inspection metadata — context captured alongside the upload.
  const [inspectionMeta, setInspectionMeta] = useState({
    projectName: "",
    inspector: "",
    siteLocation: "",
    inspectionType: "Construction Quality",
  });

  // Inspection ID assigned once analysis completes successfully.
  const [inspectionId, setInspectionId] = useState("");

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
    }, STAGE_INTERVAL_MS);

    return () => clearInterval(interval);
  }, [loading]);

  /* ── INSPECTION METADATA HELPERS ── */
  const updateMeta = useCallback((field, value) => {
    setInspectionMeta(prev => ({ ...prev, [field]: value }));
  }, []);

  /* ── FILE HELPERS ──
     Validates incoming files (type, count, size, duplicates) before
     creating previews, so bad input is rejected with a clear message
     instead of silently entering the gallery.
  ── */
  const addFiles = useCallback((selected) => {
    const valid = Array.from(selected).filter(f => f.type.startsWith("image/"));
    if (!valid.length) return;

    setFiles(prev => {
      // Maximum count check (against what's already uploaded)
      if (prev.length + valid.length > MAX_IMAGES) {
        setError(`Maximum ${MAX_IMAGES} images allowed.`);
        return prev;
      }

      // Maximum file size check
      const oversized = valid.find(
        file => file.size > MAX_FILE_SIZE_MB * 1024 * 1024
      );
      if (oversized) {
        setError(`${oversized.name} exceeds ${MAX_FILE_SIZE_MB} MB.`);
        return prev;
      }

      // Duplicate prevention (by filename against existing uploads)
      const existingNames = new Set(prev.map(f => f.file.name));
      const uniqueFiles = valid.filter(file => !existingNames.has(file.name));

      if (!uniqueFiles.length) {
        setError("These image(s) have already been uploaded.");
        return prev;
      }

      const entries = uniqueFiles.map(f => ({
        id: crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2),
        file: f,
        preview: URL.createObjectURL(f),
      }));

      const next = [...prev, ...entries];
      setActiveIdx(next.length - 1);
      return next;
    });

    setError("");
    setResult(null);
  }, []);

  const removeFile = useCallback((id) => {
    setFiles(prev => {
      const next = prev.filter(f => f.id !== id);
      setActiveIdx(i => Math.min(i, Math.max(0, next.length - 1)));
      return next;
    });
  }, []);

  /* ── UPLOAD STATISTICS ── */
  const totalSize = useMemo(() => {
    return files.reduce((sum, f) => sum + f.file.size, 0);
  }, [files]);

  const formatBytes = useCallback((bytes) => {
    if (!bytes) return "0 MB";
    return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
  }, []);

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
  const analyze = useCallback(async () => {
    if (!files.length) {
      setError("Upload at least one inspection image before analysis.");
      return;
    }

    // Required metadata check — keeps every inspection record complete
    // for downstream PDF/report generation and audit readiness.
    if (!inspectionMeta.projectName.trim()) {
      setError("Project Name is required.");
      return;
    }
    if (!inspectionMeta.inspector.trim()) {
      setError("Inspector name is required.");
      return;
    }
    if (!inspectionMeta.siteLocation.trim()) {
      setError("Site Location is required.");
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

      // Guard against backend regressions / shape drift before we
      // start reading data.images[...] downstream.
      if (!Array.isArray(data.images)) {
        throw new Error("Unexpected server response.");
      }

      // The backend now returns { success, images: [...], ... }
      // Persist the full response so the report page has all image results.
      localStorage.setItem("qualityData", JSON.stringify(data));
      // Use the active image's preview as the primary preview.
      localStorage.setItem(
        "qualityImagePreview",
        files[activeIdx]?.preview || files[0]?.preview || ""
      );

      // Persist inspection metadata captured alongside this upload.
      localStorage.setItem("inspectionMeta", JSON.stringify(inspectionMeta));

      // Assign and persist a unique inspection ID for this completed run.
      const newInspectionId = `INF-${Date.now()}`;
      localStorage.setItem("inspectionId", newInspectionId);
      setInspectionId(newInspectionId);

      setResult(data);
    } catch (err) {
      console.error("[UploadImages] Analysis failed", err);
      setError(err?.message || "Enterprise inspection service unavailable.");
    } finally {
      setLoading(false);
      setStageProgress(0);
    }
  }, [files, activeIdx, inspectionMeta]);

  /* ── RESET ──
     Clears uploaded files/results AND the inspection metadata form +
     any persisted localStorage keys, so a reset doesn't leave stale
     data behind for the next inspection run.
  ── */
  const reset = useCallback(() => {
    files.forEach(f => URL.revokeObjectURL(f.preview));
    setFiles([]);
    setActiveIdx(0);
    setResult(null);
    setError("");
    setInspectionId("");
    setInspectionMeta({
      projectName: "",
      inspector: "",
      siteLocation: "",
      inspectionType: "Construction Quality",
    });
    localStorage.removeItem("qualityData");
    localStorage.removeItem("inspectionMeta");
    localStorage.removeItem("inspectionId");
    localStorage.removeItem("qualityImagePreview");
    if (inputRef.current) inputRef.current.value = "";
  }, [files]);

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

  /* ── MULTI-IMAGE INSPECTION SUMMARY ──
     Aggregates findings across every successfully processed image
     in result.images, counted by severity. Falls back to zero
     counts if a given image has no report or unrecognized severity.
  ── */
  const inspectionSummary = useMemo(() => {
    const images = result?.images || [];
    const counts = { critical: 0, high: 0, medium: 0, low: 0 };
    let totalFindings = 0;

    images.forEach((img) => {
      (img?.report || []).forEach((finding) => {
        totalFindings += 1;
        const sev = String(finding?.severity || finding?.risk || "").toLowerCase();
        if (sev.includes("crit")) counts.critical += 1;
        else if (sev.includes("high")) counts.high += 1;
        else if (sev.includes("med")) counts.medium += 1;
        else if (sev.includes("low")) counts.low += 1;
      });
    });

    return {
      imagesProcessed: images.length,
      totalFindings,
      ...counts,
    };
  }, [result]);

  /* ── TOP RISK CATEGORIES ──
     Tallies finding categories/titles across all images and
     returns the most frequent ones. Returns an empty list rather
     than guessing if no categorized data is present.
  ── */
  const topRiskCategories = useMemo(() => {
    const images = result?.images || [];
    const tally = {};

    images.forEach((img) => {
      (img?.report || []).forEach((finding) => {
        const label = finding?.category || finding?.title || finding?.issue;
        if (!label) return;
        tally[label] = (tally[label] || 0) + 1;
      });
    });

    return Object.entries(tally)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([label, count]) => ({ label, count }));
  }, [result]);

  /* ── ACTIVE PREVIEW ── */
  const activePreview = files[activeIdx]?.preview || "";

  /* ── OBJECT URL CLEANUP ──
     Revokes preview blob URLs on unmount (or whenever the files array
     changes identity) so leaving the page without hitting Reset
     doesn't leak memory.
  ── */
  useEffect(() => {
    return () => {
      files.forEach(file => {
        URL.revokeObjectURL(file.preview);
      });
    };
  }, [files]);

  /* ── INSPECTION READINESS CHECKLIST ── */
  const readiness = useMemo(() => {
    const hasImages = files.length > 0;
    const allSupported = hasImages && files.every(f => f.file.type.startsWith("image/"));
    return [
      { label: "Images Uploaded", ready: hasImages },
      { label: "Supported Format", ready: allSupported },
      { label: "AI Engine Ready", ready: hasImages },
      { label: "Report Generation Ready", ready: hasImages },
    ];
  }, [files]);

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

        {/* ── INSPECTION METADATA ── */}
        <PageCard className="inspection-meta-card" style={{ marginBottom: 16 }}>
          <div className="report-section-header">
            <FaIdBadge /> Inspection Metadata
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
              gap: 12,
              marginTop: 12,
            }}
          >
            <label style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: "0.85rem" }}>
              Project Name <span style={{ color: "#e5484d" }}>*</span>
              <input
                type="text"
                className="minimal-input"
                placeholder="e.g. Riverside Tower Phase 2"
                value={inspectionMeta.projectName}
                onChange={(e) => updateMeta("projectName", e.target.value)}
                required
              />
            </label>
            <label style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: "0.85rem" }}>
              Inspector <span style={{ color: "#e5484d" }}>*</span>
              <input
                type="text"
                className="minimal-input"
                placeholder="Inspector name"
                value={inspectionMeta.inspector}
                onChange={(e) => updateMeta("inspector", e.target.value)}
                required
              />
            </label>
            <label style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: "0.85rem" }}>
              Site / Location <span style={{ color: "#e5484d" }}>*</span>
              <input
                type="text"
                className="minimal-input"
                placeholder="e.g. Block C, Level 4"
                value={inspectionMeta.siteLocation}
                onChange={(e) => updateMeta("siteLocation", e.target.value)}
                required
              />
            </label>
            <label style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: "0.85rem" }}>
              Inspection Type
              <select
                className="minimal-input"
                value={inspectionMeta.inspectionType}
                onChange={(e) => updateMeta("inspectionType", e.target.value)}
              >
                <option>Construction Quality</option>
                <option>Safety Compliance</option>
                <option>Structural Integrity</option>
                <option>Pre-Handover</option>
              </select>
            </label>
          </div>
        </PageCard>

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
                    <div
                      className="image-meta"
                      style={{
                        fontSize: "0.65rem",
                        opacity: 0.8,
                        marginTop: 4,
                        lineHeight: 1.3,
                        textAlign: "center",
                      }}
                    >
                      <div
                        title={f.file.name}
                        style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                      >
                        {f.file.name}
                      </div>
                      <div>
                        {formatBytes(f.file.size)} · {(f.file.type.split("/")[1] || "image").toUpperCase()}
                      </div>
                    </div>
                  </div>
                ))}
                {/* Add more button */}
                {files.length < MAX_IMAGES && (
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
                )}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Upload statistics */}
          {files.length > 0 && (
            <PageCard className="upload-stats-card" style={{ marginBottom: 16 }}>
              <div className="report-section-header">
                <FaLayerGroup /> Inspection Assets
              </div>
              <div
                style={{
                  display: "flex",
                  flexWrap: "wrap",
                  gap: 24,
                  marginTop: 10,
                  alignItems: "center",
                }}
              >
                <div>
                  <strong>{files.length}</strong>
                  <div style={{ fontSize: "0.8rem", opacity: 0.75 }}>Images Uploaded</div>
                </div>
                <div>
                  <strong>{formatBytes(totalSize)}</strong>
                  <div style={{ fontSize: "0.8rem", opacity: 0.75 }}>Total Size</div>
                </div>
                <div>
                  <strong title={files[0]?.file.name}>
                    {files[0]?.file.name?.length > MAX_FILENAME_LENGTH
                      ? `${files[0].file.name.slice(0, TRUNCATED_FILENAME_LENGTH)}...`
                      : files[0]?.file.name}
                  </strong>
                  <div style={{ fontSize: "0.8rem", opacity: 0.75 }}>Primary Image</div>
                </div>
                <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 6, color: "var(--b)" }}>
                  <FaCheckCircle /> Inspection Ready
                </div>
              </div>
            </PageCard>
          )}

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

          {/* Enterprise analysis timeline */}
          <AnimatePresence>
            {loading && (
              <motion.div
                className="analysis-timeline-card"
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                style={{ marginTop: 12, marginBottom: 16 }}
              >
                <PageCard>
                  {[
                    "Upload Complete",
                    "AI Analysis",
                    "Risk Assessment",
                    "Compliance Review",
                    "Report Compilation",
                  ].map((label, i) => {
                    const stepThreshold = (i / 4) * 100;
                    const isDone = stageProgress > stepThreshold;
                    const isCurrent =
                      stageProgress >= stepThreshold &&
                      stageProgress < ((i + 1) / 4) * 100;
                    return (
                      <div
                        key={label}
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: 8,
                          padding: "4px 0",
                          fontSize: "0.85rem",
                          fontWeight: isCurrent ? 600 : 400,
                          opacity: isDone || isCurrent ? 1 : 0.45,
                          color: isDone ? "var(--b)" : "inherit",
                        }}
                      >
                        {isDone ? <FaCheckCircle /> : <FaCircle style={{ fontSize: "0.6rem" }} />}
                        {label}
                      </div>
                    );
                  })}
                </PageCard>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Inspection readiness */}
          {files.length > 0 && !result && (
            <PageCard className="readiness-card" style={{ marginBottom: 16 }}>
              <div className="report-section-header">
                <FaShieldAlt /> Inspection Readiness
              </div>
              <div
                style={{
                  display: "flex",
                  flexWrap: "wrap",
                  gap: 14,
                  marginTop: 10,
                }}
              >
                {readiness.map((r, i) => (
                  <div
                    key={i}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 6,
                      fontSize: "0.85rem",
                      color: r.ready ? "var(--b)" : "inherit",
                      opacity: r.ready ? 1 : 0.5,
                    }}
                  >
                    {r.ready ? <FaCheckCircle /> : <FaCircle style={{ fontSize: "0.6rem" }} />}
                    {r.label}
                  </div>
                ))}
              </div>
            </PageCard>
          )}

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
              {/* Inspection ID badge */}
              {inspectionId && (
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    marginBottom: 14,
                    fontSize: "0.85rem",
                    opacity: 0.85,
                  }}
                >
                  <FaIdBadge />
                  <span>Inspection ID</span>
                  <strong style={{ letterSpacing: "0.02em" }}>{inspectionId}</strong>
                </div>
              )}

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

              {/* Multi-image inspection summary */}
              {inspectionSummary.imagesProcessed > 1 && (
                <PageCard className="executive-summary-card">
                  <div className="report-section-header">
                    <FaFileImage /> Inspection Summary
                  </div>
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
                      gap: 12,
                      marginTop: 10,
                    }}
                  >
                    <div>
                      <strong>{inspectionSummary.imagesProcessed}</strong>
                      <div style={{ fontSize: "0.8rem", opacity: 0.75 }}>Images Processed</div>
                    </div>
                    <div>
                      <strong>{inspectionSummary.totalFindings}</strong>
                      <div style={{ fontSize: "0.8rem", opacity: 0.75 }}>Total Findings</div>
                    </div>
                    <div>
                      <strong style={{ color: "#e5484d" }}>{inspectionSummary.critical}</strong>
                      <div style={{ fontSize: "0.8rem", opacity: 0.75 }}>Critical</div>
                    </div>
                    <div>
                      <strong style={{ color: "#f59e0b" }}>{inspectionSummary.high}</strong>
                      <div style={{ fontSize: "0.8rem", opacity: 0.75 }}>High</div>
                    </div>
                    <div>
                      <strong>{inspectionSummary.medium}</strong>
                      <div style={{ fontSize: "0.8rem", opacity: 0.75 }}>Medium</div>
                    </div>
                    <div>
                      <strong>{inspectionSummary.low}</strong>
                      <div style={{ fontSize: "0.8rem", opacity: 0.75 }}>Low</div>
                    </div>
                  </div>
                </PageCard>
              )}

              {/* Top risk categories */}
              {topRiskCategories.length > 0 && (
                <PageCard className="executive-summary-card">
                  <div className="report-section-header">
                    <FaExclamationTriangle /> Top Risk Categories
                  </div>
                  <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 8 }}>
                    {topRiskCategories.map((rc, i) => (
                      <div
                        key={i}
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          fontSize: "0.9rem",
                        }}
                      >
                        <span>{rc.label}</span>
                        <span style={{ opacity: 0.7 }}>{rc.count}</span>
                      </div>
                    ))}
                  </div>
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
                  gap: 12,
                  flexWrap: "wrap",
                  marginTop: 20,
                }}
              >
                <button
                  className="secondary-btn"
                  onClick={() => navigate("/quality/report")}
                >
                  Skip to Report
                </button>

                <div style={{ display: "flex", gap: 12 }}>
                  <button
                    className="secondary-btn"
                    onClick={() => navigate("/quality/report")}
                  >
                    <FaFileAlt /> Generate Report →
                  </button>

                  <button
                    className="primary-btn"
                    onClick={() => navigate("/quality/best-practices")}
                  >
                    Continue to Best Practices →
                  </button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

      </div>
    </PageLayout>
  );
}