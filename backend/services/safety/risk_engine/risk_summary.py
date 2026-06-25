from typing import List, Dict, Any, Tuple

from ai_engine.utils.logger import get_logger
from backend.services.risk_engine.rules import (
    CLASS_ID_MAP,
    SITE_PROFILES,
    LABEL_NORMALIZE,
    compute_severity,
    compute_iou,
    normalize_class_name,
)


class RiskEngine:


    def __init__(
        self,
        site_type: str = "construction",
        iou_threshold: float = 0.2
    ):
        self.logger = get_logger("RiskEngine")

        if site_type not in SITE_PROFILES:
            raise ValueError(f"Unsupported site_type: {site_type}")

        self.site_type      = site_type
        profile             = SITE_PROFILES[site_type]
        self.critical_ppe   = profile["critical_ppe"]
        self.important_ppe  = profile["important_ppe"]
        self.machine_classes = profile["machine_classes"]
        self.required_ppe   = self.critical_ppe | self.important_ppe
        self.iou_threshold  = iou_threshold

    # -----------------------------------------------------
    # PPE association
    # Uses canonical names from rules.normalize_class_name()
    # and matches by class_name (normalized), not class_id.
    # -----------------------------------------------------

    def _check_ppe(
        self,
        person_box: List[float],
        detections: List[Dict]
    ) -> Tuple[List[str], Dict[str, str]]:
       
        missing_items: List[str] = []
        ppe_status:    Dict[str, str] = {}

        for ppe_name in sorted(self.required_ppe):

            matched = any(
                normalize_class_name(det["class_name"]) == ppe_name
                and det.get("bbox") is not None
                and compute_iou(person_box, det["bbox"]) > self.iou_threshold
                for det in detections
            )

            if matched:
                ppe_status[ppe_name] = "OK"
            else:
                ppe_status[ppe_name] = "MISSING"
                missing_items.append(ppe_name)

        return missing_items, ppe_status

    # -----------------------------------------------------
    # Risk Level Classification
    # -----------------------------------------------------

    @staticmethod
    def _risk_level(score: int) -> str:
        if score >= 150:
            return "HIGH"
        if score >= 70:
            return "MEDIUM"
        if score > 0:
            return "LOW"
        return "SAFE"

    # -----------------------------------------------------
    # Compliance percentage helper
    # -----------------------------------------------------

    @staticmethod
    def _compliance_pct(compliant: int, total: int) -> float:
        if total == 0:
            return 100.0
        return round(compliant / total * 100, 1)

    # -----------------------------------------------------
    # Main Evaluation
    # -----------------------------------------------------

    def evaluate(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
       
        # Identify persons by canonical name
        persons = [
            d for d in detections
            if normalize_class_name(d.get("class_name", "")) == "person"
            and d.get("bbox") is not None
        ]

        # Count equipment and structural defects
        equipment_count = sum(
            1 for d in detections
            if normalize_class_name(d.get("class_name", "")) in self.machine_classes
        )
        crack_count = sum(
            1 for d in detections
            if normalize_class_name(d.get("class_name", "")) == "crack"
        )

        # Per-PPE compliance counters  {ppe_name: {"ok": int, "missing": int}}
        ppe_compliance: Dict[str, Dict[str, int]] = {
            name: {"ok": 0, "missing": 0}
            for name in sorted(self.required_ppe)
        }

        person_reports = []
        total_risk_score = 0
        compliant_count  = 0

        for idx, person in enumerate(persons):
            person_box = person["bbox"]

            missing_items, ppe_status = self._check_ppe(person_box, detections)

            # Determine per-person risk level from missing critical vs important
            missing_critical  = [p for p in missing_items if p in self.critical_ppe]
            missing_important = [p for p in missing_items if p in self.important_ppe]

            if missing_critical:
                person_risk = "HIGH"
            elif missing_important:
                person_risk = "MEDIUM"
            else:
                person_risk = "SAFE"

            risk_score, severity = compute_severity(person_risk)
            total_risk_score    += risk_score

            compliant = len(missing_items) == 0
            if compliant:
                compliant_count += 1

            # Update per-PPE counters
            for ppe_name, status in ppe_status.items():
                if status == "OK":
                    ppe_compliance[ppe_name]["ok"] += 1
                else:
                    ppe_compliance[ppe_name]["missing"] += 1

            person_reports.append({
                "person_id":    idx,
                "track_id":     person.get("track_id"),
                "bbox":         person_box,
                "compliant":    compliant,
                "missing":      missing_items,
                "risk_level":   person_risk,
                "risk_score":   risk_score,
                "severity":     severity,
                "details":      ppe_status,
            })

        total_persons    = len(persons)
        non_compliant    = total_persons - compliant_count
        overall_risk     = self._risk_level(total_risk_score)
        _, overall_sev   = compute_severity(overall_risk)

        # Build per-PPE compliance summary
        ppe_summary = {
            name: {
                "compliant":   counts["ok"],
                "missing":     counts["missing"],
                "compliance_pct": self._compliance_pct(counts["ok"], total_persons),
            }
            for name, counts in ppe_compliance.items()
        }

        report = {
            "summary": {
                "site_type":         self.site_type,
                "total_workers":     total_persons,
                "compliant":         compliant_count,
                "non_compliant":     non_compliant,
                "compliance_pct":    self._compliance_pct(compliant_count, total_persons),
                "equipment_count":   equipment_count,
                "crack_count":       crack_count,
                "risk_score":        total_risk_score,
                "risk_level":        overall_risk,
                "severity":          overall_sev,
                "ppe_compliance":    ppe_summary,
            },
            "persons": person_reports,
        }

        self.logger.info(
            f"Risk summary | site={self.site_type} | workers={total_persons} "
            f"| compliant={compliant_count} | score={total_risk_score} "
            f"| level={overall_risk}"
        )

        return report