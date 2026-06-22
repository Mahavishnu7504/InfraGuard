import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  FaDownload,
  FaArrowLeft,
  FaShieldAlt,
  FaCheckCircle,
  FaExclamationTriangle,
  FaSpinner,
  FaFileAlt,
  FaBrain,
  FaBolt,
  FaCamera,
  FaImage,
} from "react-icons/fa";
import { motion } from "framer-motion";

import PageLayout from "../../components/PageLayout";
import PageCard from "../../components/PageCard";
import { API_BASE } from "../../services/api";

import "../quality.css";

/* ── Workflow progress ─────────────────────────────────────────── */
function WorkflowProgress({ current = 3 }) {
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

/* ── Severity → CSS class ──────────────────────────────────────── */
function severityClass(sev) {
  const s = (sev || "").toLowerCase();
  if (s === "critical") return "sev-critical";
  if (s === "high") return "sev-high";
  if (s === "medium") return "sev-medium";
  return "sev-low";
}

/* ── Per-image findings block ──────────────────────────────────── */
/*
 * Each entry in data.images[] (backend multi-image shape) looks like:
 * {
 *   image_index, image_label, image_path, annotated_image_path,
 *   compliance_score, overall_risk, inspection_grade, overall_status,
 *   executive_summary,
 *   report: [ { issue_type, severity, observation, confidence,
 *                category, recommendation, ... } ]
 * }
 *
 * We also accept the alternate key "findings" in case the backend
 * varies between image_reports[].findings and images[].report.
 */
function ImageBlock({ imgData, index }) {
  const findings = imgData?.report || imgData?.findings || [];
  const label = imgData?.image_label || `Image ${index + 1}`;
  const score = imgData?.compliance_score ?? "—";
  const risk = imgData?.overall_risk || "—";
  const grade = imgData?.inspection_grade || "—";
  const status = imgData?.overall_status || "—";
  const summary = imgData?.executive_summary || "";

  // annotated_image_path is a server-side path — not directly usable in
  // an <img> tag unless the backend also returns a data-URL or public URL.
  // We expose it here only when it starts with "http" or "data:".
  const annotatedUrl =
    imgData?.annotated_image_url ||
    (typeof imgData?.annotated_image_path === "string" &&
      (imgData.annotated_image_path.startsWith("http") ||
        imgData.annotated_image_path.startsWith("data:"))
      ? imgData.annotated_image_path
      : null);

  // The original image preview is stored separately in localStorage by
  // UploadImages, keyed as qualityImagePreview (first/active image only).
  // For per-image previews we look for an optional image_preview_url field
  // that callers may attach after analysis.
  const originalUrl =
    imgData?.image_preview_url ||
    imgData?.image_url ||
    (typeof imgData?.image_path === "string" &&
      (imgData.image_path.startsWith("http") ||
        imgData.image_path.startsWith("data:"))
      ? imgData.image_path
      : null);

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.07 }}
      style={{ marginBottom: 28 }}
    >
      {/* Image header */}
      <div className="report-section-header" style={{ marginBottom: 12 }}>
        <FaImage />
        {label}
      </div>

      {/* KPI strip for this image */}
      <div className="executive-grid" style={{ marginBottom: 16 }}>
        <PageCard className="executive-card">
          <FaShieldAlt />
          <div><h3>{score}/100</h3><span>Compliance Score</span></div>
        </PageCard>
        <PageCard className="executive-card">
          <FaCheckCircle />
          <div><h3>{grade}</h3><span>Inspection Grade</span></div>
        </PageCard>
        <PageCard className="executive-card">
          <FaExclamationTriangle />
          <div><h3>{risk}</h3><span>Risk Level</span></div>
        </PageCard>
        <PageCard className="executive-card">
          <FaFileAlt />
          <div><h3>{findings.length}</h3><span>Findings</span></div>
        </PageCard>
      </div>

      {/* Image previews */}
      {(originalUrl || annotatedUrl) && (
        <div className="report-layout" style={{ marginBottom: 16 }}>
          {originalUrl && (
            <PageCard className="report-visual-card">
              <div className="report-section-header" style={{ marginBottom: 8 }}>
                <FaBolt /> Original Image
              </div>
              <img src={originalUrl} className="report-image" alt={`${label} original`} />
            </PageCard>
          )}
          {annotatedUrl && (
            <PageCard className="report-visual-card">
              <div className="report-section-header" style={{ marginBottom: 8 }}>
                <FaBrain /> AI Annotated
              </div>
              <img src={annotatedUrl} className="report-image" alt={`${label} annotated`} />
            </PageCard>
          )}
        </div>
      )}

      {/* Executive summary for this image */}
      {summary && (
        <PageCard className="executive-summary-card" style={{ marginBottom: 16 }}>
          <div className="report-section-header" style={{ marginBottom: 8 }}>
            <FaBrain /> Summary
          </div>
          <p className="executive-summary-text">{summary}</p>
        </PageCard>
      )}

      {/* Findings list */}
      <PageCard className="minimal-report-card">
        <div className="report-section-header" style={{ marginBottom: 12 }}>
          <FaFileAlt /> Findings
        </div>

        {findings.length === 0 ? (
          <p className="executive-summary-text" style={{ color: "var(--text-muted)" }}>
            No quality deviations detected in this image.
          </p>
        ) : (
          <div className="enterprise-findings">
            {findings.map((item, i) => (
              <div key={i} className="enterprise-finding">
                <div className="finding-top">
                  <h3>{(item?.issue_type || "Finding").replaceAll("_", " ")}</h3>
                  <div className={`severity-chip ${severityClass(item?.severity)}`}>
                    {item?.severity}
                  </div>
                </div>
                <p>{item?.observation}</p>
                {item?.recommendation && (
                  <p style={{ marginTop: 6, fontStyle: "italic", opacity: 0.82 }}>
                    {item.recommendation}
                  </p>
                )}
                <div className="finding-meta">
                  <span>Confidence: {item?.confidence}</span>
                  <span>Category: {item?.category}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </PageCard>
    </motion.div>
  );
}

/* ────────────────────────────────────────────────────────────────
   ROOT CAUSE FIX SUMMARY
   ─────────────────────────────────────────────────────────────────
   Old code assumed a flat, single-image response shape:
     data.report[]            ← per-image findings
     data.compliance_score    ← top-level score
     data.analytics.*         ← nested analytics keys

   New multi-image backend wraps everything inside data.images[]:
     data.images[i].report[]          ← per-image findings
     data.images[i].compliance_score  ← per-image score
     data.total_findings              ← aggregate total

   Bugs fixed:
   1. hasData check read data.report (always []) → page always showed
      empty state. Now we check data.images?.length > 0.
   2. All KPI fields now read from the correct top-level keys returned
      by the new backend (compliance_score, inspection_grade, etc.)
      which ARE present at the top level as aggregates.
   3. Findings section replaced with per-image blocks that each show:
      original image, annotated image, findings, score, risk, recommendations.
   ──────────────────────────────────────────────────────────────── */

export default function GenerateReport() {
  const navigate = useNavigate();
  const [downloading, setDownloading] = useState(false);

  /* ── Load persisted data ── */
  const data = useMemo(() => {
    try { return JSON.parse(localStorage.getItem("qualityData") || "{}"); } catch { return {}; }
  }, []);

  /*
   * The new response shape from the multi-image backend:
   *
   * {
   *   success: true,
   *   images: [ { image_index, image_label, compliance_score, overall_risk,
   *               inspection_grade, overall_status, executive_summary,
   *               report: [...findings] } ],
   *   compliance_score: <aggregate>,
   *   inspection_grade: <aggregate>,
   *   overall_risk:     <aggregate>,
   *   overall_status:   <aggregate>,
   *   executive_summary: <aggregate>,
   *   total_images_processed: N,
   *   total_findings: N,
   *   analytics: { ... }
   * }
   *
   * hasData: true when at least one image result is present OR when the
   * backend returns a top-level compliance_score (handles both v1 & v2).
   */
  const images = data?.images || [];
  const hasData = images.length > 0 || (data?.compliance_score ?? 0) > 0;

  /* Top-level aggregate fields (populated by backend for multi-image runs) */
  const summary = data?.executive_summary || "Inspection completed.";
  const totalImages = data?.total_images_processed || images.length || 0;
  const totalFinds = data?.total_findings || images.reduce((n, img) => n + (img?.report?.length || img?.findings?.length || 0), 0);

  /* PDF via backend endpoint */
  const downloadReport = async () => {
    setDownloading(true);
    try {
      const response = await fetch(`${API_BASE}/quality/report/download`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });

      if (!response.ok) throw new Error(`Server error: ${response.status}`);

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "InfraGuard_Quality_Report.pdf";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Failed to download report:", err);
    } finally {
      setDownloading(false);
    }
  };

  /* ── EMPTY STATE ── */
  if (!hasData) {
    return (
      <PageLayout
        badge="Report Preview"
        title="Report Preview"
        subtitle="Run an inspection to generate your report."
      >
        <div className="minimal-page">
          <WorkflowProgress current={3} />

          <div className="empty-state" style={{ marginTop: 40 }}>
            <FaFileAlt />
            <h3>No Inspection Data Found</h3>
            <p>
              Run an AI inspection first to generate your executive report.
              Upload construction site images and complete the analysis workflow.
            </p>
            <button
              className="primary-btn"
              style={{ marginTop: 16 }}
              onClick={() => navigate("/quality/upload")}
            >
              <FaCamera /> Start Inspection
            </button>
          </div>
        </div>
      </PageLayout>
    );
  }

  /* ── FULL REPORT ── */
  return (
    <PageLayout
      badge="Report Preview"
      title="Report Preview"
      subtitle="Review findings and download the inspection report."
    >
      <div className="minimal-page">

        {/* PROGRESS */}
        <WorkflowProgress current={3} />

        {/* BACK */}
        <div style={{ marginBottom: 16 }}>
          <button
            className="secondary-btn"
            onClick={() => navigate("/quality/best-practices")}
          >
            <FaArrowLeft /> Back
          </button>
        </div>

        {/* HERO */}
        <motion.div
          className="enterprise-hero"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="hero-left">
            <div className="report-badge">
              <span className="live-dot" />
              InfraGuard Enterprise AI
            </div>
            <h1>Report Preview</h1>
            <p>
              Review findings and download the inspection report.
            </p>
          </div>
        </motion.div>

        {/* EXECUTIVE SUMMARY */}
        <PageCard className="executive-summary-card" style={{ marginBottom: 20 }}>
          <div className="report-section-header">
            <FaBrain /> Executive Intelligence Summary
          </div>
          <p className="executive-summary-text">{summary}</p>
          <div className="summary-metrics">
            <div>
              <span>Images</span>
              <h4>{totalImages}</h4>
            </div>
            <div>
              <span>Total Findings</span>
              <h4>{totalFinds}</h4>
            </div>
          </div>
        </PageCard>

        <div style={{ marginBottom: 24 }}>
          <div className="report-section-header" style={{ marginBottom: 16 }}>
            <FaFileAlt /> Image-Wise Inspection Results
          </div>

          {images.length > 0 ? (
            images.map((imgData, i) => (
              <ImageBlock key={i} imgData={imgData} index={i} />
            ))
          ) : (
            /* Fallback: single-image legacy data that may still carry a
               top-level report[] array (v1 backward compatibility) */
            (data?.report || []).length > 0 && (
              <ImageBlock
                imgData={{
                  image_label: "Inspection Image",
                  compliance_score: data.compliance_score,
                  overall_risk: data.overall_risk,
                  inspection_grade: data.inspection_grade,
                  overall_status: data.overall_status,
                  executive_summary: data.executive_summary,
                  report: data.report,
                }}
                index={0}
              />
            )
          )}
        </div>

        {/* NAVIGATION */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginTop: 24,
          }}
        >
          <button
            className="secondary-btn"
            onClick={() => navigate("/quality/best-practices")}
          >
            <FaArrowLeft /> Previous
          </button>

          <button
            className="primary-btn"
            onClick={downloadReport}
            disabled={downloading}
          >
            {downloading ? <FaSpinner className="spin" /> : <FaDownload />}
            {downloading ? "Generating..." : "Download PDF"}
          </button>
        </div>

      </div>
    </PageLayout>
  );
}