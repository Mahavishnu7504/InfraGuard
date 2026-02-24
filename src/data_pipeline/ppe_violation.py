def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter / (areaA + areaB - inter)


def detect_ppe_violations(detections, iou_threshold=0.2):
    persons, helmets, vests = [], [], []

    for d in detections:
        if d["class"] == "person":
            persons.append(d["box"])
        elif d["class"] == "helmet":
            helmets.append(d["box"])
        elif d["class"] == "vest":
            vests.append(d["box"])

    results = []

    for idx, p in enumerate(persons):
        has_helmet = any(iou(p, h) > iou_threshold for h in helmets)
        has_vest = any(iou(p, v) > iou_threshold for v in vests)

        if not has_helmet and not has_vest:
            risk = "HIGH"
        elif not has_helmet:
            risk = "HIGH"
        elif not has_vest:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        results.append({
            "person_id": idx,
            "helmet": has_helmet,
            "vest": has_vest,
            "risk": risk
        })

    return results