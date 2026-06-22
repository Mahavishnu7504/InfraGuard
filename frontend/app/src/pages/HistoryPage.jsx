import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
    FaHistory, FaShieldAlt, FaClock,
    FaDatabase, FaSearch, FaFilter,
} from "react-icons/fa";

import PageLayout from "../components/PageLayout";
import "./history.css";

function PDot() {
    return <span className="pdot" />;
}

const ITEM_CLS = { HIGH: "hp__item--high", MEDIUM: "hp__item--medium", LOW: "hp__item--low" };
const itemCls = (r) => ITEM_CLS[(r || "LOW").toUpperCase()] || ITEM_CLS.LOW;

export default function HistoryPage() {
    const [search, setSearch] = useState("");
    const [riskFilter, setRiskFilter] = useState("ALL");

    /* ── same demo data ─────────────────────────────────── */
    const incidents = [
        { title: "PPE Compliance Event", risk: "HIGH", time: "09:42 AM" },
        { title: "Zone Movement Detection", risk: "MEDIUM", time: "10:12 AM" },
        { title: "Inspection Audit Event", risk: "LOW", time: "11:26 AM" },
        { title: "Unauthorized Access", risk: "HIGH", time: "01:20 PM" },
        { title: "Worker Detection", risk: "LOW", time: "02:11 PM" },
    ];

    const filteredData = useMemo(() => {
        return incidents.filter(item => {
            const matchesSearch = item.title.toLowerCase().includes(search.toLowerCase());
            const matchesRisk = riskFilter === "ALL" ? true : item.risk === riskFilter;
            return matchesSearch && matchesRisk;
        });
    }, [search, riskFilter]); // eslint-disable-line

    return (
        <PageLayout>
            <div className="hp">

                {/* HEADER */}
                <motion.div className="hp__hdr" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
                    <div>
                        <div className="hp__eyebrow"><PDot /> FORENSIC TIMELINE</div>
                        <h1 className="hp__h1"><FaHistory /> Incident<span className="hp__accent"> Intelligence</span></h1>
                        <p className="hp__sub">Enterprise forensic event timeline</p>
                    </div>
                    <div className="hp__status"><FaShieldAlt /> AI VERIFIED</div>
                </motion.div>

                {/* KPI */}
                <div className="hp__kpis">
                    {[
                        { icon: <FaDatabase />, label: "Total Records", val: "1,284" },
                        { icon: <FaShieldAlt />, label: "Verification", val: "99.2%" },
                        { icon: <FaClock />, label: "Today's Events", val: "37" },
                    ].map((k, i) => (
                        <motion.div key={i} className="hp__kpi"
                            initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: i * 0.07 }}
                        >
                            <div className="hp__kpi-icon">{k.icon}</div>
                            <div>
                                <div className="hp__kpi-val">{k.val}</div>
                                <div className="hp__kpi-label">{k.label}</div>
                            </div>
                        </motion.div>
                    ))}
                </div>

                {/* CONTROLS */}
                <div className="hp__controls">
                    <div className="hp__search">
                        <FaSearch />
                        <input
                            type="text" placeholder="Search incidents..."
                            value={search} onChange={e => setSearch(e.target.value)}
                        />
                    </div>
                    <div className="hp__filter">
                        <FaFilter />
                        <select value={riskFilter} onChange={e => setRiskFilter(e.target.value)}>
                            <option>ALL</option>
                            <option>HIGH</option>
                            <option>MEDIUM</option>
                            <option>LOW</option>
                        </select>
                    </div>
                </div>

                {/* TIMELINE */}
                <div className="hp__timeline">
                    {filteredData.length > 0
                        ? filteredData.map((item, i) => (
                            <motion.div
                                key={i}
                                className={`hp__item ${itemCls(item.risk)}`}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: i * 0.05 }}
                            >
                                <div className="hp__item-dot" />
                                <div className="hp__item-main">
                                    <h3>{item.title}</h3>
                                    <p>AI monitored event recorded successfully.</p>
                                </div>
                                <div className="hp__item-side">
                                    <span className="hp__risk">{item.risk}</span>
                                    <small>{item.time}</small>
                                </div>
                            </motion.div>
                        ))
                        : <div className="hp__empty">No matching incidents found</div>
                    }
                </div>

            </div>
        </PageLayout>
    );
}