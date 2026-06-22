import cv2

COLOR_MAP = {
    "low": (0, 255, 0),
    "medium": (0, 165, 255),
    "high": (0, 0, 255),
}

def draw_detections(frame, detections):
    for det in detections:
        x = det["x"]
        y = det["y"]
        w = det["w"]
        h = det["h"]

        label = det["label"]
        risk = det["risk"]

        color = COLOR_MAP.get(risk, (255, 255, 255))

        # draw box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # draw label
        cv2.putText(
            frame,
            f"{label} ({risk})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    return frame