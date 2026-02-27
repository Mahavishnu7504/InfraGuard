from src.utils.logger import get_logger


class RiskEngine:
    """
    Industry-grade Risk Engine
    - Per-person PPE validation
    - Bounding box overlap matching
    - Structured risk report
    """

    def __init__(self):
        self.logger = get_logger("RiskEngine")

    @staticmethod
    def _iou(boxA, boxB):
        """
        Compute Intersection over Union between two boxes
        Boxes format: [x1, y1, x2, y2]
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_area = max(0, xB - xA) * max(0, yB - yA)

        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        union_area = boxA_area + boxB_area - inter_area

        if union_area == 0:
            return 0

        return inter_area / union_area

    def evaluate(self, detections):
        """
        Evaluate PPE compliance per person
        Class IDs:
            0 -> person
            1 -> helmet
            2 -> vest
        """

        persons = [d for d in detections if d["class_id"] == 0]
        helmets = [d for d in detections if d["class_id"] == 1]
        vests = [d for d in detections if d["class_id"] == 2]

        report = {
            "total_persons": len(persons),
            "violations": [],
            "severity": "LOW"
        }

        for idx, person in enumerate(persons):
            person_box = person["bbox"]

            has_helmet = any(
                self._iou(person_box, helmet["bbox"]) > 0.2
                for helmet in helmets
            )

            has_vest = any(
                self._iou(person_box, vest["bbox"]) > 0.2
                for vest in vests
            )

            person_violation = {
                "person_id": idx,
                "helmet": "OK" if has_helmet else "MISSING",
                "vest": "OK" if has_vest else "MISSING"
            }

            if not has_helmet or not has_vest:
                report["violations"].append(person_violation)

        if len(report["violations"]) > 0:
            report["severity"] = "HIGH"

        self.logger.info(f"Risk report generated: {report}")

        return report