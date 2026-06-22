from typing import List, Dict, Any

from ai_engine.utils.logger import get_logger
from backend.services.risk_engine.rules import (
    CLASS_ID_MAP,
    SITE_PROFILES,
    compute_severity
)


class RiskEngine:
    """
    InfraGuard Risk Engine

    Responsibilities
    ----------------
    • Detect PPE compliance
    • Associate PPE with persons using IoU
    • Compute risk score and severity
    • Generate structured risk report
    """

    def __init__(
        self,
        site_type: str = "construction",
        iou_threshold: float = 0.2
    ):

        self.logger = get_logger("RiskEngine")

        if site_type not in SITE_PROFILES:
            raise ValueError(f"Unsupported site_type: {site_type}")

        self.site_type = site_type
        self.required_ppe = SITE_PROFILES[site_type]
        self.iou_threshold = iou_threshold

    # -----------------------------------------------------
    # IoU computation
    # -----------------------------------------------------

    @staticmethod
    def _compute_iou(boxA, boxB) -> float:

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_area = max(0, xB - xA) * max(0, yB - yA)

        if inter_area == 0:
            return 0.0

        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        union_area = boxA_area + boxB_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    # -----------------------------------------------------
    # PPE association
    # -----------------------------------------------------

    def _check_ppe(self, person_box, detections):

        missing_items = []
        ppe_status = {}

        for ppe_name in self.required_ppe:

            class_id = CLASS_ID_MAP[ppe_name]

            matched = any(
                det.get("class_id") == class_id
                and det.get("bbox") is not None
                and self._compute_iou(person_box, det["bbox"]) > self.iou_threshold
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
    # Main Evaluation
    # -----------------------------------------------------

    def evaluate(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:

        persons = [
            d for d in detections
            if d.get("class_id") == CLASS_ID_MAP["person"]
        ]

        report = {
            "summary": {
                "site_type": self.site_type,
                "total_persons": len(persons),
                "compliant": 0,
                "non_compliant": 0,
                "risk_score": 0,
                "risk_level": "SAFE"
            },
            "persons": []
        }

        total_risk_score = 0

        for idx, person in enumerate(persons):

            person_box = person.get("bbox")

            if person_box is None:
                continue

            missing_items, ppe_status = self._check_ppe(person_box, detections)

            risk_score, severity = compute_severity(missing_items)

            total_risk_score += risk_score

            compliant = len(missing_items) == 0

            if compliant:
                report["summary"]["compliant"] += 1
            else:
                report["summary"]["non_compliant"] += 1

            report["persons"].append({
                "person_id": idx,
                "bbox": person_box,
                "compliant": compliant,
                "missing": missing_items,
                "risk_score": risk_score,
                "severity": severity,
                "details": ppe_status
            })

        report["summary"]["risk_score"] = total_risk_score
        report["summary"]["risk_level"] = self._risk_level(total_risk_score)

        self.logger.info(
            f"Risk summary generated | Persons: {len(persons)} | Score: {total_risk_score}"
        )

        return report