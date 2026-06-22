import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
    FaExclamationTriangle, FaShieldAlt,
    FaCheckCircle, FaClock, FaBroadcastTower,
} from "react-icons/fa";

import PageLayout from "../components/PageLayout";
import { getAlerts } from "../services/api";
import "./alertCenter.css";

function PDot({ color = "#00ff9d" }) {
    return <span className="pdot" style={{ "--c": color }} />;
}

const CARD_CLS = { HIGH: "al__card--high", MEDIUM: "al__card--medium", LOW: "al__card--low" };
const cardCls = (r) => CARD_CLS[(r || "LOW").toUpperCase()] || CARD_CLS.LOW;

export default function AlertCenter() {
    const [alerts, setAlerts] = useState([]);
    const [filter, setFilter] = useState("ALL");

    /* ── load (unchanged) ──────────────────────────────── */
    useEffect(() => {
        loadAlerts();
        const iv = setInterval(loadAlerts, 4000);
        return () => clearInterval(iv);
    }, []);

    const loadAlerts = async () => {
        try {
            const data = await getAlerts();
            setAlerts(data || []);
        } catch (err) {
            console.error("[ALERT ERROR]", err);
        }
    };

    /* ── filter (unchanged) ────────────────────────────── */
    const filteredAlerts = filter === "ALL"
        ? alerts
        : alerts.filter(item => item.risk_level?.toUpperCase() === filter);

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
                    <div className="al__status"><PDot /> LIVE ALERT ENGINE</div>
                </motion.div>

                {/* FILTERS */}
                <div className="al__filters">
                    {["ALL", "HIGH", "MEDIUM", "LOW"].map((level, i) => (
                        <motion.button
                            key={i}
                            className={`al__filter-btn${filter === level ? ` al__filter-btn--active al__filter-btn--${level}` : ""}`}
                            whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }}
                            onClick={() => setFilter(level)}
                        >
                            {level}
                        </motion.button>
                    ))}
                </div>

                {/* GRID */}
                <div className="al__grid">
                    {filteredAlerts.length > 0
                        ? filteredAlerts.map((alert, i) => (
                            <motion.div
                                key={i}
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
                                        <FaClock /> {alert.timestamp || "LIVE"}
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
                                <span>No active alerts</span>
                            </div>
                        )
                    }
                </div>

            </div>
        </PageLayout>
    );
}