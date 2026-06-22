import { useNavigate } from "react-router-dom";
import { useState } from "react";
import {
  FaArrowRight, FaCheckCircle,
  FaInfoCircle, FaTimes, FaFileAlt,
} from "react-icons/fa";
import { motion, AnimatePresence } from "framer-motion";

import PageLayout from "../components/PageLayout";
import "./quality.css";

/* ── Workflow progress indicator (reused context: step 0 = landing) */
function WorkflowProgress({ current = 0 }) {
  const steps = [
    { label: "Overview" },
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
            {i > 0 && (
              <div className={`wp-connector ${isDone ? "done" : ""}`} />
            )}
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

function PDot() {
  return <span className="pdot-qa" />;
}

export default function QualityAssurance() {
  const navigate = useNavigate();
  const [showWorkflow, setShowWorkflow] = useState(false);

  const workflow = [
    { step: "01", title: "Upload Image", desc: "Drag & drop or select inspection images" },
    { step: "02", title: "AI Analysis", desc: "Vision model scans for defects & risks" },
    { step: "03", title: "Review Findings", desc: "Per-image findings, severity & compliance score" },
    { step: "04", title: "Best Practices", desc: "Issue-specific corrective & preventive actions" },
    { step: "05", title: "Executive Report", desc: "PDF-ready findings with AI recommendations" },
  ];

  return (
    <PageLayout>
      <div className="qa qa--compact">

        {/* ── HEADER ── */}
        <motion.div
          className="quality-topbar"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <div>
            <div className="qa__eyebrow"><PDot /> Quality Intelligence</div>
            <h1>Quality Assurance</h1>
          </div>
          <div className="quality-status"><PDot /> AI Active</div>
        </motion.div>

        {/* ── WORKFLOW PROGRESS ── */}
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}>
          <WorkflowProgress current={0} />
        </motion.div>

        {/* ── HERO ── */}
        <motion.div
          className="quality-hero quality-hero--centered"
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className="quality-hero-left">
            <h2>Construction Inspection Intelligence</h2>
            <p>
              AI-powered inspection and compliance reporting, from image to
              executive summary.
            </p>
            <div className="quality-actions">
              <button
                className="quality-primary-btn"
                onClick={() => navigate("/quality/upload")}
              >
                Start Inspection <FaArrowRight />
              </button>
              <button
                className="quality-secondary-btn"
                onClick={() => navigate("/quality/report")}
              >
                <FaFileAlt /> Reports
              </button>
              <button
                className="quality-secondary-btn"
                onClick={() => setShowWorkflow(true)}
              >
                <FaInfoCircle /> How it works
              </button>
            </div>
          </div>
        </motion.div>

        {/* ── WORKFLOW MODAL ── */}
        <AnimatePresence>
          {showWorkflow && (
            <motion.div
              className="workflow-overlay"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowWorkflow(false)}
            >
              <motion.div
                className="workflow-modal"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ duration: 0.26, ease: [0.22, 1, 0.36, 1] }}
                onClick={e => e.stopPropagation()}
              >
                <div className="workflow-modal-header">
                  <h2>Inspection Workflow</h2>
                  <button className="close-btn" onClick={() => setShowWorkflow(false)}>
                    <FaTimes />
                  </button>
                </div>
                <div className="workflow-grid">
                  {workflow.map((item, i) => (
                    <div key={i} className="workflow-item">
                      <span>{item.step}</span>
                      <h3>{item.title}</h3>
                      <p style={{ fontFamily: "var(--mono)", fontSize: "0.62rem", color: "var(--tx2)", marginTop: 6, lineHeight: 1.7 }}>
                        {item.desc}
                      </p>
                    </div>
                  ))}
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

      </div>
    </PageLayout>
  );
}