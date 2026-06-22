def analyze_ppe(detections):

    violations = []

    for d in detections:
        if d["class_id"] == 0:
            violations.append("Person detected")

    return violations