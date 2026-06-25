import { useMemo } from "react";
import {
  FaCheckCircle,
  FaExclamationTriangle,
  FaBrain,
  FaArrowRight,
  FaArrowLeft,
  FaUserShield,
  FaClipboardCheck,
  FaChartBar,
  FaTable,
  FaBullseye,
} from "react-icons/fa";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";

import PageLayout from "../../components/PageLayout";
import PageCard from "../../components/PageCard";

import "../quality.css";

/* ── Workflow progress ─────────────────────────────────────────── */
function WorkflowProgress({ current = 2 }) {
  const steps = [
    { label: "Upload" },
    { label: "Findings" },
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

/* ── Flatten findings from multi-image response ───────────────────
   ROOT CAUSE FIX:
   The old code read findings from qualityData.report — a top-level
   array that existed in the single-image v1 response.

   The new multi-image backend nests findings inside:
     qualityData.images[i].report[]

   We flatten all per-image findings into a single array so the
   existing buildAIPractices() logic works without other changes.

   Backward-compat: if data.report[] still exists (legacy v1 shape),
   we fall back to it so old stored data also renders correctly.
─────────────────────────────────────────────────────────────────── */
function extractFindings(qualityData) {
  // New multi-image shape: data.images[].report[]
  if (Array.isArray(qualityData?.images) && qualityData.images.length > 0) {
    return qualityData.images.flatMap(img => img?.report || img?.findings || []);
  }
  // Legacy single-image shape: data.report[]
  if (Array.isArray(qualityData?.report) && qualityData.report.length > 0) {
    return qualityData.report;
  }
  return [];
}

/* ── PRIORITY CLASSIFICATION ──
   Maps a finding category to a severity tier used for the
   priority badge on each recommendation card.
─────────────────────────────────────────────────────────────────── */
function getPriority(category = "") {
  const c = category.toLowerCase();
  if (c.includes("rebar") || c.includes("crack")) return "Critical";
  if (c.includes("housekeeping") || c.includes("ppe") || c.includes("material") || c.includes("damage")) return "High";
  return "Medium";
}

/* ── RESPONSIBLE TEAM ASSIGNMENT ──
   Maps a finding category to the team accountable for remediation.
─────────────────────────────────────────────────────────────────── */
function getResponsibleTeam(category = "") {
  const c = category.toLowerCase();
  if (c.includes("structural") || c.includes("rebar") || c.includes("crack")) return "Structural Engineer";
  if (c.includes("housekeeping")) return "Site Supervisor";
  if (c.includes("ppe") || c.includes("safety")) return "Safety Officer";
  if (c.includes("material") || c.includes("damage")) return "Quality Inspector";
  return "Site Supervisor";
}

/* ── COMPLIANCE STANDARDS MAPPING ──
   Maps a finding category to the applicable regulatory/quality
   standards referenced in the recommendation card.
─────────────────────────────────────────────────────────────────── */
function getComplianceMapping(category = "") {
  const c = category.toLowerCase();
  if (c.includes("ppe") || c.includes("safety")) return ["OSHA 1926", "ISO 45001"];
  if (c.includes("housekeeping")) return ["OSHA Site Safety"];
  if (c.includes("structural") || c.includes("rebar") || c.includes("crack")) return ["Construction QA", "ISO 9001"];
  if (c.includes("material") || c.includes("damage")) return ["ISO 9001"];
  return ["General Site Compliance"];
}

/* ── RISK IMPACT ──
   Maps a finding category to a plain-language description of
   the operational consequence if left unaddressed.
─────────────────────────────────────────────────────────────────── */
function getRiskImpact(category = "") {
  const c = category.toLowerCase();
  if (c.includes("housekeeping")) return "Trip hazards and operational disruption.";
  if (c.includes("ppe") || c.includes("safety")) return "Worker injury exposure.";
  if (c.includes("crack")) return "Structural degradation.";
  if (c.includes("rebar")) return "Corrosion and integrity loss.";
  if (c.includes("material") || c.includes("damage")) return "Asset deterioration and rework costs.";
  return "Operational and compliance risk if unaddressed.";
}

/* ── PRIORITY BADGE COLORS ──
   Static lookup — defined once at module scope so it isn't
   recreated on every render of the AI practices grid.
─────────────────────────────────────────────────────────────────── */
const PRIORITY_COLORS = {
  Critical: "#e5484d",
  High: "#f59e0b",
  Medium: "#3b82f6",
  Low: "#6b7280",
};

/* ── REPORT READINESS CHECKLIST ──
   Static labels for the Report Readiness panel.
─────────────────────────────────────────────────────────────────── */
const REPORT_CHECKLIST = [
  "Findings Reviewed",
  "Recommendations Generated",
  "Corrective Actions Identified",
  "Ready For Report Generation",
];

/* ── Derive AI-contextual practices from analysis results ─────── */
function buildAIPractices(findings = []) {
  if (!findings.length) return [];

  const unique = [...new Set(findings.map(f => f?.category).filter(Boolean))];
  return unique.slice(0, 3).map(cat => {
    const related = findings.filter(f => f?.category === cat);
    return {
      title: `${cat.replaceAll("_", " ")} — AI Recommendations`,
      icon: <FaBrain />,
      description: `Derived from ${related.length} AI finding${related.length > 1 ? "s" : ""} in this inspection. Address these to improve your compliance score.`,
      aiGenerated: true,
      points: related
        .slice(0, 4)
        .map(f => f?.recommendation || f?.observation || "Review and remediate this finding."),
    };
  });
}

export default function BestPractices() {
  const navigate = useNavigate();

  /* Read persisted analysis if available */
  const qualityData = useMemo(() => {
    const stored = localStorage.getItem("qualityData");
    if (!stored) return {};
    try { return JSON.parse(stored); } catch { return {}; }
  }, []);

  // FIX: flatten findings from data.images[].report[] (new shape)
  //      instead of reading data.report[] (old shape, now always empty)
  const aiFindings = useMemo(() => extractFindings(qualityData), [qualityData]);
  const hasAnalysis = aiFindings.length > 0;
  const aiPractices = useMemo(() => buildAIPractices(aiFindings), [aiFindings]);

  /* ── INSPECTION SUMMARY ──
     Aggregates aiFindings by severity and category for the
     summary panel. Read-only derivation; does not alter
     extractFindings() or buildAIPractices().
  ── */
  const inspectionSummary = useMemo(() => {
    const counts = { critical: 0, high: 0 };
    const categories = new Set();

    aiFindings.forEach((f) => {
      const sev = String(f?.severity || f?.risk || "").toLowerCase();
      if (sev.includes("crit")) counts.critical += 1;
      else if (sev.includes("high")) counts.high += 1;
      if (f?.category) categories.add(f.category);
    });

    return {
      total: aiFindings.length,
      critical: counts.critical,
      high: counts.high,
      categoriesAffected: categories.size,
      recommendationsGenerated: aiPractices.length,
    };
  }, [aiFindings, aiPractices]);

  /* ── TOP RISK AREAS ──
     Ranks the categories with the most findings, mapped to a
     readable label via getResponsibleTeam's category families.
  ── */
  const topRiskAreas = useMemo(() => {
    const tally = {};
    aiFindings.forEach((f) => {
      const cat = f?.category;
      if (!cat) return;
      tally[cat] = (tally[cat] || 0) + 1;
    });
    return Object.entries(tally)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([cat, count]) => ({
        label: cat.replaceAll("_", " "),
        count,
        priority: getPriority(cat),
      }));
  }, [aiFindings]);

  /* ── EXECUTIVE / MANAGEMENT RECOMMENDATION ──
     Plain-language summary generated from aiFindings severity
     counts, shown in the management recommendation section.
  ── */
  const managementRecommendation = useMemo(() => {
    const { critical, high, total } = inspectionSummary;
    if (!total) return "";
    if (critical > 0) {
      return `This inspection identified ${critical} critical and ${high} high-severity finding${high === 1 ? "" : "s"}. Immediate corrective action is recommended before operational continuation.`;
    }
    if (high > 0) {
      return `This inspection identified ${high} high-severity finding${high === 1 ? "" : "s"} with no critical issues. Corrective action is recommended within the standard remediation window.`;
    }
    return `This inspection identified ${total} finding${total === 1 ? "" : "s"}, none rated critical or high severity. Routine follow-up is recommended.`;
  }, [inspectionSummary]);

  /* ── RECOMMENDATION MATRIX ROWS ──
     One row per AI practice card, reusing the same category →
     priority/team/action derivation used on the cards above.
  ── */
  const matrixRows = useMemo(() => {
    return aiPractices.map((item) => {
      const category = item.title.split(" — ")[0];
      const priority = getPriority(category);
      const team = getResponsibleTeam(category);
      const action =
        priority === "Critical"
          ? "Engineering Review"
          : priority === "High"
            ? "Immediate Cleanup"
            : "Scheduled Remediation";
      return {
        issue: category.replaceAll("_", " "),
        priority,
        owner: team,
        action,
      };
    });
  }, [aiPractices]);

  return (
    <PageLayout
      badge="AI Recommendations"
      title="Best Practices"
      subtitle="Recommendations generated from inspection findings."
    >
      <div className="minimal-page">

        {/* ── PROGRESS ── */}
        <WorkflowProgress current={2} />

        {/* ── BACK ── */}
        <div style={{ marginBottom: 16 }}>
          <button
            className="secondary-btn"
            onClick={() => navigate("/quality/upload")}
          >
            <FaArrowLeft /> Back
          </button>
        </div>

        {/* ── HERO ── */}
        <motion.div
          className="enterprise-hero"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="hero-left">
            <div className="report-badge">
              <span className="live-dot" />
              Enterprise Compliance Center
            </div>
            <h1>Best Practices</h1>
            <p>
              AI-generated recommendations based on inspection findings.
            </p>
          </div>
        </motion.div>

        {/* AI-derived badge */}
        {hasAnalysis && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.1 }}
            style={{ marginBottom: 16 }}
          >
            <div className="ai-generated-badge">
              <FaBrain />
              {aiPractices.length} AI-generated recommendations from your last inspection
            </div>
          </motion.div>
        )}

        {/* ── INSPECTION SUMMARY PANEL ── */}
        {hasAnalysis && (
          <PageCard className="executive-summary-card" style={{ marginBottom: 16 }}>
            <div className="report-section-header">
              <FaChartBar /> Inspection Summary
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
                <strong>{inspectionSummary.total}</strong>
                <div style={{ fontSize: "0.8rem", opacity: 0.75 }}>Total Findings</div>
              </div>
              <div>
                <strong style={{ color: "#e5484d" }}>{inspectionSummary.critical}</strong>
                <div style={{ fontSize: "0.8rem", opacity: 0.75 }}>Critical Findings</div>
              </div>
              <div>
                <strong style={{ color: "#f59e0b" }}>{inspectionSummary.high}</strong>
                <div style={{ fontSize: "0.8rem", opacity: 0.75 }}>High Findings</div>
              </div>
              <div>
                <strong>{inspectionSummary.categoriesAffected}</strong>
                <div style={{ fontSize: "0.8rem", opacity: 0.75 }}>Categories Affected</div>
              </div>
              <div>
                <strong>{inspectionSummary.recommendationsGenerated}</strong>
                <div style={{ fontSize: "0.8rem", opacity: 0.75 }}>Recommendations Generated</div>
              </div>
            </div>
          </PageCard>
        )}

        {/* ── TOP RISK AREAS ── */}
        {hasAnalysis && topRiskAreas.length > 0 && (
          <PageCard className="executive-summary-card" style={{ marginBottom: 16 }}>
            <div className="report-section-header">
              <FaBullseye /> Highest Risk Areas
            </div>
            <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 8 }}>
              {topRiskAreas.map((area) => (
                <div
                  key={area.label}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    fontSize: "0.9rem",
                  }}
                >
                  <span style={{ textTransform: "capitalize" }}>{area.label}</span>
                  <span style={{ display: "flex", alignItems: "center", gap: 10, opacity: 0.8 }}>
                    {area.priority}
                    <span style={{ opacity: 0.6 }}>· {area.count} finding{area.count === 1 ? "" : "s"}</span>
                  </span>
                </div>
              ))}
            </div>
          </PageCard>
        )}

        {/* ── GRID ── */}
        <div className="practice-grid">
          {!hasAnalysis && (
            <PageCard className="practice-enterprise-card">
              <div className="practice-header">
                <div className="practice-icon"><FaBrain /></div>
                <div>
                  <h3>No Recommendations Yet</h3>
                  <p>Run an inspection to generate AI-derived best practices here.</p>
                </div>
              </div>
              <button
                className="primary-btn"
                style={{ marginTop: 16 }}
                onClick={() => navigate("/quality/upload")}
              >
                Go to Upload
              </button>
            </PageCard>
          )}
          {aiPractices.map((item, index) => {
            const category = item.title.split(" — ")[0];
            const priority = getPriority(category);
            const team = getResponsibleTeam(category);
            const compliance = getComplianceMapping(category);
            const riskImpact = getRiskImpact(category);

            return (
              <motion.div
                key={item.title}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.055 }}
              >
                <PageCard className="practice-enterprise-card" style={{ position: "relative" }}>

                  {/* Priority badge — top right of card */}
                  <div
                    style={{
                      position: "absolute",
                      top: 16,
                      right: 16,
                      fontSize: "0.7rem",
                      fontWeight: 700,
                      letterSpacing: "0.04em",
                      padding: "3px 10px",
                      borderRadius: 999,
                      color: "#fff",
                      background: PRIORITY_COLORS[priority] || "#6b7280",
                    }}
                  >
                    {priority.toUpperCase()}
                  </div>

                  {/* AI badge on derived cards */}
                  <div className="ai-generated-badge" style={{ marginBottom: 14 }}>
                    <FaBrain /> AI Derived
                  </div>

                  {/* Header */}
                  <div className="practice-header">
                    <div className="practice-icon">{item.icon}</div>
                    <div>
                      <h3>{item.title}</h3>
                      <p>{item.description}</p>
                    </div>
                  </div>

                  {/* Risk Impact / Priority / Responsible / Compliance */}
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
                      gap: 12,
                      margin: "14px 0",
                      padding: "12px 0",
                      borderTop: "1px solid var(--border)",
                      borderBottom: "1px solid var(--border)",
                    }}
                  >
                    <div>
                      <div style={{ fontSize: "0.72rem", opacity: 0.6, textTransform: "uppercase" }}>Risk Impact</div>
                      <div style={{ fontSize: "0.85rem", marginTop: 2 }}>{riskImpact}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: "0.72rem", opacity: 0.6, textTransform: "uppercase" }}>Priority</div>
                      <div style={{ fontSize: "0.85rem", marginTop: 2, fontWeight: 600 }}>{priority}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: "0.72rem", opacity: 0.6, textTransform: "uppercase" }}>Responsible Team</div>
                      <div style={{ fontSize: "0.85rem", marginTop: 2 }}>{team}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: "0.72rem", opacity: 0.6, textTransform: "uppercase" }}>Compliance Area</div>
                      <div style={{ fontSize: "0.85rem", marginTop: 2 }}>{compliance.join(" / ")}</div>
                    </div>
                  </div>

                  {/* Points */}
                  <div className="practice-points">
                    {item.points.map((point) => (
                      <div key={`${item.title}-${point}`} className="practice-point">
                        <FaCheckCircle />
                        <span>{point}</span>
                      </div>
                    ))}
                  </div>

                  {/* Applicable standards */}
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 12 }}>
                    {compliance.map((std, i) => (
                      <span
                        key={i}
                        style={{
                          fontSize: "0.72rem",
                          padding: "3px 9px",
                          borderRadius: 6,
                          background: "rgba(0,200,255,0.08)",
                          border: "1px solid var(--border-hi)",
                        }}
                      >
                        {std}
                      </span>
                    ))}
                  </div>

                  {/* Footer */}
                  <div className="practice-footer">
                    <FaExclamationTriangle />
                    Generated from inspection findings.
                  </div>

                </PageCard>
              </motion.div>
            );
          })}
        </div>

        {/* ── RECOMMENDATION MATRIX ── */}
        {hasAnalysis && matrixRows.length > 0 && (
          <PageCard className="executive-summary-card" style={{ marginTop: 16, overflowX: "auto" }}>
            <div className="report-section-header">
              <FaTable /> Recommendation Matrix
            </div>
            <table style={{ width: "100%", marginTop: 12, borderCollapse: "collapse", fontSize: "0.85rem" }}>
              <thead>
                <tr style={{ textAlign: "left", borderBottom: "1px solid var(--border)" }}>
                  <th style={{ padding: "8px 6px", opacity: 0.7 }}>Issue</th>
                  <th style={{ padding: "8px 6px", opacity: 0.7 }}>Priority</th>
                  <th style={{ padding: "8px 6px", opacity: 0.7 }}>Owner</th>
                  <th style={{ padding: "8px 6px", opacity: 0.7 }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {matrixRows.map((row) => (
                  <tr key={`${row.issue}-${row.owner}`} style={{ borderBottom: "1px solid var(--border)" }}>
                    <td style={{ padding: "8px 6px", textTransform: "capitalize" }}>{row.issue}</td>
                    <td style={{ padding: "8px 6px" }}>{row.priority}</td>
                    <td style={{ padding: "8px 6px" }}>{row.owner}</td>
                    <td style={{ padding: "8px 6px" }}>{row.action}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </PageCard>
        )}

        {/* ── MANAGEMENT RECOMMENDATION ── */}
        {hasAnalysis && managementRecommendation && (
          <PageCard className="executive-summary-card" style={{ marginTop: 16 }}>
            <div className="report-section-header">
              <FaUserShield /> Management Recommendation
            </div>
            <p className="executive-summary-text" style={{ marginTop: 8 }}>
              {managementRecommendation}
            </p>
          </PageCard>
        )}

        {/* ── REPORT READINESS STATUS ── */}
        {hasAnalysis && (
          <PageCard className="executive-summary-card" style={{ marginTop: 16, marginBottom: 8 }}>
            <div className="report-section-header">
              <FaClipboardCheck /> Report Readiness
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 14, marginTop: 10 }}>
              {REPORT_CHECKLIST.map((label) => (
                <div
                  key={label}
                  style={{ display: "flex", alignItems: "center", gap: 6, fontSize: "0.85rem", color: "var(--b)" }}
                >
                  <FaCheckCircle /> {label}
                </div>
              ))}
            </div>
          </PageCard>
        )}

        {/* ── NAVIGATION ── */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginTop: 24,
          }}
        >
          <button
            className="secondary-btn"
            onClick={() => navigate("/quality/upload")}
          >
            <FaArrowLeft /> Previous
          </button>

          <button
            className="primary-btn"
            onClick={() => navigate("/quality/report")}
          >
            Next <FaArrowRight />
          </button>
        </div>

      </div>
    </PageLayout>
  );
}