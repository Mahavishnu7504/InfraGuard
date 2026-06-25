

import cv2
import time


# Same risk palette style as main pipeline
RISK_COLORS = {
    "LOW": (0, 255, 120),
    "MEDIUM": (0, 215, 255),
    "HIGH": (0, 80, 255),
    "CRITICAL": (0, 0, 255),
}


CLASS_COLORS = {
    "person": (0, 200, 255),
    "helmet": (0, 255, 120),
    "vest": (0, 180, 255),
    "boots": (60, 255, 200),
    "gloves": (120, 255, 180),
    "crack": (0, 215, 255),
    "excavator": (180, 180, 255),
    "loader": (180, 180, 255),
}


def _color_for_detection(det):
    risk = str(det.get("risk", "LOW")).upper()

    if risk in RISK_COLORS:
        return RISK_COLORS[risk]

    return CLASS_COLORS.get(
        str(det.get("class_name", "")).lower(),
        (255, 255, 255)
    )


def draw_detections(frame, detections):
    """
    Main reusable drawing function.

    Supports old detections and new pipeline detections.
    """

    for det in detections:

        bbox = det.get("bbox")

        # backward compatibility with old x,y,w,h format
        if not bbox:
            x = det.get("x", 0)
            y = det.get("y", 0)
            w = det.get("w", 0)
            h = det.get("h", 0)
            bbox = [x, y, x + w, y + h]

        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(int, bbox)

        color = _color_for_detection(det)

        class_name = det.get(
            "class_name",
            det.get("label", "object")
        )

        risk = det.get("risk", "LOW")

        confidence = det.get("confidence")

        tracking_id = det.get(
            "tracking_id",
            det.get("track_id")
        )

        label = class_name

        if tracking_id is not None:
            label += f" #{tracking_id}"

        if confidence is not None:
            label += f" {float(confidence)*100:.0f}%"

        label += f" [{risk}]"


        # box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )


        # label background
        cv2.rectangle(
            frame,
            (x1, y1 - 25),
            (x1 + 220, y1),
            color,
            -1
        )


        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )


    return frame



def draw_trajectories(frame, detections):
    """
    Draw object movement history.
    """

    for det in detections:

        points = det.get("trajectory", [])

        if len(points) < 2:
            continue

        color = _color_for_detection(det)

        for i in range(1, len(points)):
            cv2.line(
                frame,
                tuple(points[i-1]),
                tuple(points[i]),
                color,
                2
            )

    return frame



def draw_danger_zones(frame, zones):
    """
    Draw danger zone polygons.
    """

    pulse = abs(__import__("math").sin(time.time() * 3))

    for zone in zones:

        polygon = zone.get("polygon")

        if not polygon:
            continue

        pts = __import__("numpy").array(
            polygon,
            dtype="int32"
        )

        overlay = frame.copy()

        cv2.fillPoly(
            overlay,
            [pts],
            (0, 0, 255)
        )

        frame[:] = cv2.addWeighted(
            overlay,
            0.15 + pulse * 0.1,
            frame,
            0.85 - pulse * 0.1,
            0
        )

        cv2.polylines(
            frame,
            [pts],
            True,
            (0, 0, 255),
            2
        )

    return frame



def draw_frame(frame, detections, zones=None):
    """
    Complete renderer.

    Intended to be called by detection_service.py.
    """

    if zones:
        draw_danger_zones(frame, zones)

    draw_detections(frame, detections)

    draw_trajectories(frame, detections)

    return frame
