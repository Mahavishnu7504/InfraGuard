import { useMemo } from "react";
import {
  FaCheckCircle,
  FaExclamationTriangle,
  FaBrain,
  FaArrowRight,
  FaArrowLeft,
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
    try { return JSON.parse(localStorage.getItem("qualityData") || "{}"); } catch { return {}; }
  }, []);

  // FIX: flatten findings from data.images[].report[] (new shape)
  //      instead of reading data.report[] (old shape, now always empty)
  const aiFindings = useMemo(() => extractFindings(qualityData), [qualityData]);
  const hasAnalysis = aiFindings.length > 0;
  const aiPractices = useMemo(() => buildAIPractices(aiFindings), [aiFindings]);

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
            </PageCard>
          )}
          {aiPractices.map((item, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.055 }}
            >
              <PageCard className="practice-enterprise-card">

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

                {/* Points */}
                <div className="practice-points">
                  {item.points.map((point, i) => (
                    <div key={i} className="practice-point">
                      <FaCheckCircle />
                      <span>{point}</span>
                    </div>
                  ))}
                </div>

                {/* Footer */}
                <div className="practice-footer">
                  <FaExclamationTriangle />
                  Generated from inspection findings.
                </div>

              </PageCard>
            </motion.div>
          ))}
        </div>

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