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
  FaIdBadge,
  FaUserTie,
  FaMapMarkerAlt,
  FaCalendarAlt,
  FaHashtag,
  FaClipboardCheck,
  FaChartPie,
  FaTable,
  FaUserShield,
  FaListOl,
  FaBalanceScale,
  FaSignature,
  FaUserClock,
} from "react-icons/fa";
import { motion } from "framer-motion";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
} from "recharts";

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

/* ── Severity → display color (for charts/badges) ─────────────── */
function severityColor(sev) {
  const s = (sev || "").toLowerCase();
  if (s === "critical") return "#dc2626";
  if (s === "high") return "#f59e0b";
  if (s === "medium") return "#3b82f6";
  return "#10b981";
}

/* ── Inspection metadata loader (PHASE 1) ──────────────────────── */
/*
 * UploadImages.jsx persists project/site/inspector context under the
 * "inspectionMeta" key. We read it defensively — if it's missing,
 * malformed, or from an older flow that never wrote it, every field
 * falls back to a placeholder instead of throwing.
 */
function loadInspectionMeta() {
  // UploadImages.jsx writes the inspection ID as a separate top-level key
  // ("inspectionId"), not inside the "inspectionMeta" blob. We merge both
  // sources here so InspectionMetadataCard always has the ID when available.
  const inspectionId = localStorage.getItem("inspectionId");

  try {
    const raw = localStorage.getItem("inspectionMeta");
    const parsed = raw ? JSON.parse(raw) || {} : {};
    return { inspectionId, ...parsed };
  } catch {
    return { inspectionId };
  }
}

/* ── Severity distribution aggregator (PHASE 3) ────────────────── */
/*
 * Walks every image's findings (images[].report[] / .findings[]) and
 * tallies counts per severity bucket. Works whether the data came from
 * the v2 multi-image shape or the v1 single-image fallback.
 */
function aggregateSeverity(images, fallbackReport) {
  const counts = { Critical: 0, High: 0, Medium: 0, Low: 0 };
  const sources =
    images && images.length > 0
      ? images.flatMap((img) => img?.report || img?.findings || [])
      : fallbackReport || [];

  sources.forEach((item) => {
    const sev = (item?.severity || "").toLowerCase();
    if (sev === "critical") counts.Critical += 1;
    else if (sev === "high") counts.High += 1;
    else if (sev === "medium") counts.Medium += 1;
    else counts.Low += 1;
  });

  return counts;
}

/* ── Top risks by issue_type frequency (PHASE 5) ───────────────── */
function computeTopRisks(images, fallbackReport, limit = 3) {
  const sources =
    images && images.length > 0
      ? images.flatMap((img) => img?.report || img?.findings || [])
      : fallbackReport || [];

  const tally = {};
  sources.forEach((item) => {
    const key = item?.issue_type || "Unclassified Issue";
    tally[key] = (tally[key] || 0) + 1;
  });

  return Object.entries(tally)
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([issue_type, count]) => ({ issue_type, count }));
}

/* ── Confidence analytics (PHASE 8) ────────────────────────────── */
/*
 * Backend's analytics_summary() may already provide aggregate confidence
 * figures under data.analytics. We prefer those when present; otherwise
 * we derive a best-effort estimate client-side from per-finding confidence.
 */
function computeConfidenceAnalytics(data, images, fallbackReport) {
  const analytics = data?.analytics || {};

  if (
    analytics.average_confidence != null ||
    analytics.high_confidence_findings != null ||
    analytics.review_required_findings != null
  ) {
    return {
      average:
        analytics.average_confidence != null
          ? `${analytics.average_confidence}%`
          : "—",
      high: analytics.high_confidence_findings ?? "—",
      review: analytics.review_required_findings ?? "—",
    };
  }

  const sources =
    images && images.length > 0
      ? images.flatMap((img) => img?.report || img?.findings || [])
      : fallbackReport || [];

  if (sources.length === 0) {
    return { average: "—", high: "—", review: "—" };
  }

  let total = 0;
  let parsedCount = 0;
  let high = 0;
  let review = 0;

  sources.forEach((item) => {
    const raw = item?.confidence;
    const num =
      typeof raw === "number"
        ? raw
        : typeof raw === "string"
          ? parseFloat(raw.replace("%", ""))
          : NaN;

    if (!Number.isNaN(num)) {
      total += num;
      parsedCount += 1;
      if (num >= 80) high += 1;
      else if (num < 60) review += 1;
    }
  });

  return {
    average: parsedCount > 0 ? `${Math.round(total / parsedCount)}%` : "—",
    high,
    review,
  };
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

                {/* PHASE 6 — Findings Intelligence upgrade.
                    These three fields are optional/back-compat: if the
                    backend hasn't started returning risk_impact /
                    responsible_team / target_resolution yet, we fall back
                    to sensible defaults rather than showing blank lines. */}
                <div
                  className="finding-meta"
                  style={{ flexWrap: "wrap", gap: "6px 18px" }}
                >
                  <span>Confidence: {item?.confidence}</span>
                  <span>Category: {item?.category}</span>
                  <span>
                    Risk Impact: {item?.risk_impact || "Under assessment"}
                  </span>
                  <span>
                    Responsible: {item?.responsible_team || "Site Engineer"}
                  </span>
                  <span>
                    Target Resolution:{" "}
                    {item?.target_resolution ||
                      (item?.severity?.toLowerCase() === "critical"
                        ? "Within 24 Hours"
                        : item?.severity?.toLowerCase() === "high"
                          ? "Within 3 Days"
                          : "Within 14 Days")}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </PageCard>
    </motion.div>
  );
}

/* ── PHASE 1 — Inspection Metadata Block ───────────────────────── */
function InspectionMetadataCard({ meta }) {
  const fields = [
    { icon: <FaHashtag />, label: "Inspection ID", value: meta?.inspectionId || meta?.inspection_id },
    { icon: <FaClipboardCheck />, label: "Project Name", value: meta?.projectName || meta?.project_name },
    { icon: <FaMapMarkerAlt />, label: "Site Location", value: meta?.siteLocation || meta?.site_location },
    { icon: <FaUserTie />, label: "Inspector", value: meta?.inspector || meta?.inspectorName },
    { icon: <FaCalendarAlt />, label: "Inspection Date", value: meta?.inspectionDate || meta?.date },
  ];

  return (
    <PageCard className="minimal-report-card" style={{ marginBottom: 20 }}>
      <div className="report-section-header" style={{ marginBottom: 14 }}>
        <FaIdBadge /> Inspection Metadata
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
          gap: "14px 20px",
        }}
      >
        {fields.map((f, i) => (
          <div key={i} style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
            <div style={{ opacity: 0.65, marginTop: 2 }}>{f.icon}</div>
            <div>
              <div style={{ fontSize: "0.75rem", opacity: 0.6, marginBottom: 2 }}>
                {f.label}
              </div>
              <div style={{ fontWeight: 600 }}>{f.value || "Not Provided"}</div>
            </div>
          </div>
        ))}
      </div>
    </PageCard>
  );
}

/* ── PHASE 2 — Executive KPI Dashboard (6 cards) ───────────────── */
function ExecutiveKpiDashboard({ data, totalImages, totalFinds }) {
  const cards = [
    {
      icon: <FaShieldAlt />,
      value: data?.compliance_score != null ? `${data.compliance_score}/100` : "—",
      label: "Compliance Score",
    },
    {
      icon: <FaCheckCircle />,
      value: data?.inspection_grade || "—",
      label: "Inspection Grade",
    },
    {
      icon: <FaExclamationTriangle />,
      value: data?.overall_risk || "—",
      label: "Overall Risk",
    },
    {
      icon: <FaUserShield />,
      value: data?.audit_readiness || "Pending Review",
      label: "Audit Readiness",
    },
    {
      icon: <FaFileAlt />,
      value: data?.total_findings ?? totalFinds,
      label: "Total Findings",
    },
    {
      icon: <FaImage />,
      value: data?.total_images_processed ?? totalImages,
      label: "Images Processed",
    },
  ];

  return (
    <div
      className="executive-grid"
      style={{
        marginBottom: 20,
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
        gap: 14,
      }}
    >
      {cards.map((c, i) => (
        <PageCard key={i} className="executive-card">
          {c.icon}
          <div>
            <h3>{c.value}</h3>
            <span>{c.label}</span>
          </div>
        </PageCard>
      ))}
    </div>
  );
}

/* ── PHASE 3 — Severity Distribution Section ───────────────────── */
function SeverityDistributionSection({ counts }) {
  const data = [
    { name: "Critical", value: counts.Critical },
    { name: "High", value: counts.High },
    { name: "Medium", value: counts.Medium },
    { name: "Low", value: counts.Low },
  ];
  const total = data.reduce((n, d) => n + d.value, 0);

  return (
    <PageCard className="minimal-report-card" style={{ marginBottom: 20 }}>
      <div className="report-section-header" style={{ marginBottom: 14 }}>
        <FaChartPie /> Severity Distribution
      </div>

      {total === 0 ? (
        <p className="executive-summary-text" style={{ color: "var(--text-muted)" }}>
          No findings recorded to distribute by severity.
        </p>
      ) : (
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            gap: 24,
          }}
        >
          <div style={{ width: 220, height: 220, flexShrink: 0 }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={data}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={45}
                  outerRadius={85}
                  paddingAngle={2}
                >
                  {data.map((d, i) => (
                    <Cell key={i} fill={severityColor(d.name)} />
                  ))}
                </Pie>
                <RechartsTooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Progress-bar style breakdown — doubles as a fallback if the
              pie chart's container has zero width on first paint. */}
          <div style={{ flex: 1, minWidth: 220 }}>
            {data.map((d, i) => {
              const pct = total > 0 ? Math.round((d.value / total) * 100) : 0;
              return (
                <div key={i} style={{ marginBottom: 10 }}>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      fontSize: "0.85rem",
                      marginBottom: 4,
                    }}
                  >
                    <span>{d.name}</span>
                    <span>
                      {d.value} ({pct}%)
                    </span>
                  </div>
                  <div
                    style={{
                      height: 8,
                      borderRadius: 4,
                      background: "rgba(255,255,255,0.08)",
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        height: "100%",
                        width: `${pct}%`,
                        background: severityColor(d.name),
                        borderRadius: 4,
                        transition: "width 0.4s ease",
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </PageCard>
  );
}

/* ── PHASE 4 — Corrective Action Matrix ────────────────────────── */
/*
 * Backend doesn't always supply owner/action/priority directly on each
 * finding yet, so we derive sensible defaults from severity when those
 * fields are absent — keeping the table populated and useful today,
 * while picking up real backend values transparently once available.
 */
function deriveOwner(item) {
  if (item?.owner) return item.owner;
  const cat = (item?.category || "").toLowerCase();
  if (cat.includes("structural")) return "Structural Engineer";
  if (cat.includes("electrical")) return "Electrical Engineer";
  if (cat.includes("safety")) return "Site Safety Officer";
  if (cat.includes("housekeeping")) return "Site Supervisor";
  return "Site Engineer";
}

function derivePriority(sev) {
  const s = (sev || "").toLowerCase();
  if (s === "critical") return "Immediate";
  if (s === "high") return "Urgent";
  if (s === "medium") return "Scheduled";
  return "Routine";
}

function CorrectiveActionMatrix({ images, fallbackReport }) {
  const findings =
    images && images.length > 0
      ? images.flatMap((img) => img?.report || img?.findings || [])
      : fallbackReport || [];

  return (
    <PageCard className="minimal-report-card" style={{ marginBottom: 20 }}>
      <div className="report-section-header" style={{ marginBottom: 14 }}>
        <FaTable /> Corrective Action Matrix
      </div>

      {findings.length === 0 ? (
        <p className="executive-summary-text" style={{ color: "var(--text-muted)" }}>
          No corrective actions required.
        </p>
      ) : (
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.88rem" }}>
            <thead>
              <tr style={{ textAlign: "left", borderBottom: "1px solid rgba(255,255,255,0.12)" }}>
                <th style={{ padding: "8px 10px" }}>Issue</th>
                <th style={{ padding: "8px 10px" }}>Severity</th>
                <th style={{ padding: "8px 10px" }}>Priority</th>
                <th style={{ padding: "8px 10px" }}>Owner</th>
                <th style={{ padding: "8px 10px" }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {findings.map((item, i) => (
                <tr key={i} style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                  <td style={{ padding: "8px 10px" }}>
                    {(item?.issue_type || "Finding").replaceAll("_", " ")}
                  </td>
                  <td style={{ padding: "8px 10px" }}>
                    <span className={`severity-chip ${severityClass(item?.severity)}`}>
                      {item?.severity}
                    </span>
                  </td>
                  <td style={{ padding: "8px 10px" }}>{derivePriority(item?.severity)}</td>
                  <td style={{ padding: "8px 10px" }}>{deriveOwner(item)}</td>
                  <td style={{ padding: "8px 10px" }}>
                    {item?.action || item?.recommendation || "Engineering Review"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </PageCard>
  );
}

/* ── PHASE 5 — Executive Risk Summary (Top 3 Risks) ────────────── */
function ExecutiveRiskSummary({ topRisks }) {
  return (
    <PageCard className="minimal-report-card" style={{ marginBottom: 20 }}>
      <div className="report-section-header" style={{ marginBottom: 14 }}>
        <FaListOl /> Top Risks
      </div>

      {topRisks.length === 0 ? (
        <p className="executive-summary-text" style={{ color: "var(--text-muted)" }}>
          No recurring risk patterns identified.
        </p>
      ) : (
        <ol style={{ margin: 0, paddingLeft: 20 }}>
          {topRisks.map((r, i) => (
            <li key={i} style={{ marginBottom: 8 }}>
              <span style={{ fontWeight: 600 }}>
                {r.issue_type.replaceAll("_", " ")}
              </span>{" "}
              <span style={{ opacity: 0.65, fontSize: "0.85rem" }}>
                ({r.count} occurrence{r.count > 1 ? "s" : ""})
              </span>
            </li>
          ))}
        </ol>
      )}
    </PageCard>
  );
}

/* ── PHASE 7 — Compliance Benchmark Panel ──────────────────────── */
function ComplianceBenchmarkPanel({ data }) {
  const benchmark = data?.compliance_benchmark || data?.benchmark || "Industry Acceptable";
  const auditStatus = data?.audit_status || "Conditionally Audit Ready";
  const operationalStatus = data?.operational_status || "Stable";

  return (
    <PageCard className="minimal-report-card" style={{ marginBottom: 20 }}>
      <div className="report-section-header" style={{ marginBottom: 14 }}>
        <FaBalanceScale /> Compliance Benchmark
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
          gap: 14,
        }}
      >
        <div>
          <div style={{ fontSize: "0.75rem", opacity: 0.6, marginBottom: 4 }}>Benchmark</div>
          <div style={{ fontWeight: 600 }}>{benchmark}</div>
        </div>
        <div>
          <div style={{ fontSize: "0.75rem", opacity: 0.6, marginBottom: 4 }}>Audit Status</div>
          <div style={{ fontWeight: 600 }}>{auditStatus}</div>
        </div>
        <div>
          <div style={{ fontSize: "0.75rem", opacity: 0.6, marginBottom: 4 }}>
            Operational Status
          </div>
          <div style={{ fontWeight: 600 }}>{operationalStatus}</div>
        </div>
      </div>
    </PageCard>
  );
}

/* ── PHASE 8 (display) — AI Confidence Analytics ───────────────── */
function ConfidenceAnalyticsPanel({ analytics }) {
  return (
    <PageCard className="minimal-report-card" style={{ marginBottom: 20 }}>
      <div className="report-section-header" style={{ marginBottom: 14 }}>
        <FaBrain /> AI Confidence Analytics
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
          gap: 14,
        }}
      >
        <div>
          <div style={{ fontSize: "0.75rem", opacity: 0.6, marginBottom: 4 }}>
            Average AI Confidence
          </div>
          <div style={{ fontWeight: 700, fontSize: "1.2rem" }}>{analytics.average}</div>
        </div>
        <div>
          <div style={{ fontSize: "0.75rem", opacity: 0.6, marginBottom: 4 }}>
            High Confidence Findings
          </div>
          <div style={{ fontWeight: 700, fontSize: "1.2rem" }}>{analytics.high}</div>
        </div>
        <div>
          <div style={{ fontSize: "0.75rem", opacity: 0.6, marginBottom: 4 }}>
            Review Required Findings
          </div>
          <div style={{ fontWeight: 700, fontSize: "1.2rem" }}>{analytics.review}</div>
        </div>
      </div>
    </PageCard>
  );
}

/* ── PHASE 9 — Report Readiness Section ────────────────────────── */
function ReportReadinessSection({ hasFindings, hasImages }) {
  const checks = [
    { label: "Inspection Completed", done: hasImages },
    { label: "Findings Validated", done: hasFindings },
    { label: "Recommendations Generated", done: hasFindings },
    { label: "PDF Ready", done: hasImages },
  ];

  return (
    <PageCard className="minimal-report-card" style={{ marginBottom: 20 }}>
      <div className="report-section-header" style={{ marginBottom: 14 }}>
        <FaClipboardCheck /> Report Readiness
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 16 }}>
        {checks.map((c, i) => (
          <div
            key={i}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              opacity: c.done ? 1 : 0.45,
            }}
          >
            <FaCheckCircle style={{ color: c.done ? "#10b981" : "inherit" }} />
            <span>{c.label}</span>
          </div>
        ))}
      </div>
    </PageCard>
  );
}

/* ── PHASE 10 — Management Sign-Off Preview ────────────────────── */
function ManagementSignOffPreview() {
  const rows = [
    { icon: <FaUserTie />, label: "Prepared By" },
    { icon: <FaUserClock />, label: "Reviewed By" },
    { icon: <FaUserShield />, label: "Approved By" },
  ];

  return (
    <PageCard className="minimal-report-card" style={{ marginBottom: 20 }}>
      <div className="report-section-header" style={{ marginBottom: 14 }}>
        <FaSignature /> Management Sign-Off
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
          gap: 18,
        }}
      >
        {rows.map((r, i) => (
          <div key={i}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, opacity: 0.7, marginBottom: 6 }}>
              {r.icon}
              <span style={{ fontSize: "0.8rem" }}>{r.label}</span>
            </div>
            <div
              style={{
                height: 1,
                background: "rgba(255,255,255,0.15)",
                marginTop: 26,
              }}
            />
          </div>
        ))}
        <div>
          <div style={{ fontSize: "0.8rem", opacity: 0.7, marginBottom: 6 }}>Signature</div>
          <div
            style={{
              height: 1,
              background: "rgba(255,255,255,0.15)",
              marginTop: 26,
            }}
          />
        </div>
      </div>
    </PageCard>
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
  const [downloadError, setDownloadError] = useState("");

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

  /* PHASE 1 — Inspection metadata persisted by UploadImages.jsx */
  const inspectionMeta = useMemo(() => loadInspectionMeta(), []);

  /* PHASE 3 — Severity distribution across all images (or v1 fallback) */
  const severityCounts = useMemo(
    () => aggregateSeverity(images, data?.report),
    [images, data?.report]
  );

  /* PHASE 5 — Top recurring risks by issue_type frequency */
  const topRisks = useMemo(
    () => computeTopRisks(images, data?.report, 3),
    [images, data?.report]
  );

  /* PHASE 8 — Aggregate AI confidence analytics */
  const confidenceAnalytics = useMemo(
    () => computeConfidenceAnalytics(data, images, data?.report),
    [data, images]
  );

  /* PDF via backend endpoint */
  const downloadReport = async () => {
    setDownloading(true);
    setDownloadError("");
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
      a.download = `InfraGuard_Report_${inspectionMeta?.inspectionId || Date.now()}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Failed to download report:", err);
      setDownloadError(
        "PDF generation failed. Please try again or contact support if the issue persists."
      );
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

        {/* PHASE 1 — Inspection Metadata Block */}
        <InspectionMetadataCard meta={inspectionMeta} />

        {/* PHASE 2 — Executive KPI Dashboard (6 cards) */}
        <ExecutiveKpiDashboard data={data} totalImages={totalImages} totalFinds={totalFinds} />

        {/* EXECUTIVE SUMMARY */}
        <div className="report-generated" style={{ marginBottom: 8, fontSize: "0.8rem", opacity: 0.65 }}>
          Generated: {new Date().toLocaleString()}
        </div>
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

        {/* PHASE 3 — Severity Distribution Section */}
        <SeverityDistributionSection counts={severityCounts} />

        {/* PHASE 5 — Executive Risk Summary (Top 3 Risks) */}
        <ExecutiveRiskSummary topRisks={topRisks} />

        {/* Zero-findings callout — purely additive, doesn't affect data flow */}
        {totalFinds === 0 && (
          <PageCard className="minimal-report-card" style={{ marginBottom: 20 }}>
            <div className="report-section-header" style={{ marginBottom: 8 }}>
              <FaCheckCircle /> No Findings Detected
            </div>
            <p className="executive-summary-text" style={{ color: "var(--text-muted)" }}>
              Site appears compliant.
            </p>
          </PageCard>
        )}

        {/* PHASE 4 — Corrective Action Matrix */}
        <CorrectiveActionMatrix images={images} fallbackReport={data?.report} />

        {/* PHASE 7 — Compliance Benchmark Panel */}
        <ComplianceBenchmarkPanel data={data} />

        {/* PHASE 8 — AI Confidence Analytics */}
        <ConfidenceAnalyticsPanel analytics={confidenceAnalytics} />

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

        {/* PHASE 9 — Report Readiness Section */}
        <ReportReadinessSection hasFindings={totalFinds > 0} hasImages={hasData} />

        {/* PHASE 10 — Management Sign-Off Preview */}
        <ManagementSignOffPreview />

        {/* NAVIGATION */}
        {downloadError && (
          <div className="error-box" style={{ marginTop: 16 }}>
            {downloadError}
          </div>
        )}
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