import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
    FaExclamationTriangle, FaShieldAlt,
    FaCheckCircle, FaClock, FaBroadcastTower,
    FaSyncAlt, FaRedo,
} from "react-icons/fa";

import PageLayout from "../components/PageLayout";
import { getAlerts } from "../services/api";
import "./alertCenter.css";

const REFRESH_INTERVAL = 4000;

function PDot({ color = "#00ff9d" }) {
    return <span className="pdot" style={{ "--c": color }} />;
}

const CARD_CLS = { HIGH: "al__card--high", MEDIUM: "al__card--medium", LOW: "al__card--low" };
const cardCls = (r) => CARD_CLS[(r || "LOW").toUpperCase()] || CARD_CLS.LOW;

const formatTime = (ts) => {
    if (!ts) return "LIVE";
    const d = new Date(ts);
    return Number.isNaN(d.getTime()) ? ts : d.toLocaleTimeString();
};

export default function AlertCenter() {
    const [alerts, setAlerts] = useState([]);
    const [filter, setFilter] = useState("ALL");
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [lastUpdated, setLastUpdated] = useState(null);

    /* ── load ───────────────────────────────────────────── */
    useEffect(() => {
        loadAlerts();
        const iv = setInterval(loadAlerts, REFRESH_INTERVAL);
        return () => clearInterval(iv);
    }, []);

    const loadAlerts = async () => {
        try {
            const data = await getAlerts();
            setAlerts(data || []);
            setError(null);
            setLastUpdated(new Date());
        } catch (err) {
            console.error("[ALERT ERROR]", err);
            setError("Unable to load alerts.");
        } finally {
            setLoading(false);
        }
    };

    const handleRetry = () => {
        setLoading(true);
        setError(null);
        loadAlerts();
    };

    /* ── filter ─────────────────────────────────────────── */
    const filteredAlerts = useMemo(() => {
        return filter === "ALL"
            ? alerts
            : alerts.filter(item => item.risk_level?.toUpperCase() === filter);
    }, [alerts, filter]);

    const filterCounts = useMemo(() => {
        const counts = { ALL: alerts.length, HIGH: 0, MEDIUM: 0, LOW: 0 };
        alerts.forEach((a) => {
            const lvl = a.risk_level?.toUpperCase();
            if (lvl && counts[lvl] !== undefined) counts[lvl] += 1;
        });
        return counts;
    }, [alerts]);

    /* ── loading state ──────────────────────────────────── */
    if (loading) {
        return (
            <PageLayout>
                <div className="al">
                    <div className="al__loading">
                        <FaSyncAlt className="al__loading-icon" />
                        <span>Loading Alerts...</span>
                    </div>
                </div>
            </PageLayout>
        );
    }

    /* ── error state ────────────────────────────────────── */
    if (error) {
        return (
            <PageLayout>
                <div className="al">
                    <div className="al__error">
                        <FaExclamationTriangle className="al__error-icon" />
                        <span>{error}</span>
                        <button type="button" className="al__retry-btn" onClick={handleRetry}>
                            <FaRedo /> Retry
                        </button>
                    </div>
                </div>
            </PageLayout>
        );
    }

    return (
        <PageLayout>
            <div className="al">

                {/* HEADER */}
                <motion.div className="al__hdr" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    <div>
                        <div className="al__eyebrow"><PDot /> ALERT INTELLIGENCE</div>
                        <h1 className="al__h1">Alert<span className="al__accent"> Center</span></h1>
                        <p className="al__sub">Realtime AI incident escalation monitoring</p>
                    </div>
                    <div className="al__hdr-right">
                        <div className="al__status"><PDot /> LIVE ALERT ENGINE</div>
                        <div className="al__meta-row">
                            <span className="al__count">{alerts.length} Active Alerts</span>
                            {lastUpdated && (
                                <span className="al__updated">
                                    Last Updated {lastUpdated.toLocaleTimeString()}
                                </span>
                            )}
                            <span className="al__refresh-rate">
                                Refresh: Every {REFRESH_INTERVAL / 1000}s
                            </span>
                        </div>
                    </div>
                </motion.div>

                {/* FILTERS */}
                <div className="al__filters">
                    {["ALL", "HIGH", "MEDIUM", "LOW"].map((level) => (
                        <motion.button
                            key={level}
                            className={`al__filter-btn${filter === level ? ` al__filter-btn--active al__filter-btn--${level}` : ""}`}
                            whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }}
                            onClick={() => setFilter(level)}
                        >
                            {level} <span className="al__filter-count">({filterCounts[level] ?? 0})</span>
                        </motion.button>
                    ))}
                </div>

                {/* GRID */}
                <div className="al__grid">
                    {filteredAlerts.length > 0
                        ? filteredAlerts.map((alert, i) => (
                            <motion.div
                                key={alert.id ?? `${alert.timestamp}-${alert.event_type}`}
                                className={`al__card ${cardCls(alert.risk_level)}`}
                                initial={{ opacity: 0, y: 18 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: i * 0.04 }}
                            >
                                <div className="al__card-top">
                                    <div className="al__risk-badge">
                                        <FaExclamationTriangle /> {alert.risk_level}
                                    </div>
                                    <div className="al__time">
                                        <FaClock /> {formatTime(alert.timestamp)}
                                    </div>
                                </div>

                                <div className="al__card-body">
                                    <h3>{alert.event_type}</h3>
                                    <p>{alert.description}</p>
                                </div>

                                <div className="al__footer">
                                    <div className="al__chip"><FaShieldAlt /> AI VERIFIED</div>
                                    <div className="al__chip al__chip--secondary"><FaBroadcastTower /> ACTIVE</div>
                                </div>
                            </motion.div>
                        ))
                        : (
                            <div className="al__empty">
                                <FaCheckCircle />
                                <span className="al__empty-title">No Active Alerts</span>
                                <span className="al__empty-sub">All monitored locations are operating safely.</span>
                            </div>
                        )
                    }
                </div>

            </div>
        </PageLayout>
    );
}