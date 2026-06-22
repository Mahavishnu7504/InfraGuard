# =========================================================
# INFRA GUARD — ENTERPRISE SAFETY INTELLIGENCE ENGINE
# =========================================================

from typing import Dict, Any

import cv2
import time
import math
import numpy as np
import traceback

from datetime import datetime

from ai_engine.pipelines.safety_pipeline import (
    run_safety_pipeline
)

# =========================================================
# DANGER ZONES
# =========================================================

DANGER_ZONES = [

    {

        "name":
            "CRANE ZONE",

        "risk":
            "CRITICAL",

        "polygon": [

            [820, 180],
            [1180, 180],
            [1240, 520],
            [780, 520]
        ]
    },

    {

        "name":
            "MACHINE AREA",

        "risk":
            "HIGH",

        "polygon": [

            [120, 420],
            [420, 420],
            [420, 690],
            [120, 690]
        ]
    }
]

# =========================================================
# PPE INTELLIGENCE
# =========================================================

PPE_VIOLATIONS = {

    "no_helmet":
        "HIGH",

    "no_vest":
        "MEDIUM",

    "no_gloves":
        "LOW"
}

# =========================================================
# RISK SCORE
# =========================================================

RISK_SCORES = {

    "critical": 90,

    "high": 70,

    "medium": 45,

    "low": 20
}

# =========================================================
# ENTERPRISE ANALYTICS
# =========================================================

def calculate_operational_risk(
    detections
):

    score = 0

    highest = "LOW"

    severity_order = {

        "LOW": 0,
        "MEDIUM": 1,
        "HIGH": 2,
        "CRITICAL": 3
    }

    for det in detections:

        risk = str(
            det.get(
                "risk",
                "LOW"
            )
        ).upper()

        score += RISK_SCORES.get(
            risk.lower(),
            10
        )

        if severity_order[risk] > severity_order[highest]:

            highest = risk

    score = min(score, 100)

    return {

        "risk_score":
            score,

        "overall_risk":
            highest
    }

# =========================================================
# AI METADATA
# =========================================================

def build_ai_metadata():

    return {

        "engine":
            "InfraGuard Enterprise AI",

        "pipeline":
            "Realtime Safety Intelligence",

        "analysis_mode":
            "Operational Surveillance",

        "timestamp":
            datetime.utcnow().isoformat()
    }

# =========================================================
# INCIDENT CLASSIFICATION
# =========================================================

def classify_detection(det):

    label = str(
        det.get(
            "class_name",
            ""
        )
    ).lower()

    confidence = float(
        det.get(
            "confidence",
            0
        )
    )

    # =============================================
    # DEFAULT
    # =============================================

    det["risk"] = "LOW"

    det["priority"] = 1

    det["incident_type"] = (
        "Operational Observation"
    )

    # =============================================
    # PPE
    # =============================================

    for violation, risk in PPE_VIOLATIONS.items():

        if violation in label:

            det["risk"] = risk

            det["priority"] = 3

            det["incident_type"] = (
                "PPE Non Compliance"
            )

    # =============================================
    # CRACK
    # =============================================

    if "crack" in label:

        det["risk"] = "MEDIUM"

        det["priority"] = 2

        det["incident_type"] = (
            "Infrastructure Degradation"
        )

    # =============================================
    # PERSON
    # =============================================

    if "person" in label:

        det["incident_type"] = (
            "Personnel Activity"
        )

    # =============================================
    # CONFIDENCE
    # =============================================

    if confidence >= 0.90:

        det["confidence_level"] = (
            "Enterprise Verified"
        )

    elif confidence >= 0.80:

        det["confidence_level"] = (
            "High Confidence"
        )

    else:

        det["confidence_level"] = (
            "Review Recommended"
        )

    return det

# =========================================================
# DANGER ZONE INTELLIGENCE
# =========================================================

def analyze_intrusions(
    detections
):

    alerts = []

    for det in detections:

        label = str(
            det.get(
                "class_name",
                ""
            )
        ).lower()

        bbox = det.get(
            "bbox",
            []
        )

        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox

        cx = int((x1 + x2) / 2)

        cy = int((y1 + y2) / 2)

        if "person" in label:

            for zone in DANGER_ZONES:

                polygon = zone[
                    "polygon"
                ]

                inside = (

                    cv2.pointPolygonTest(

                        np.array(
                            polygon,
                            dtype=np.int32
                        ),

                        (cx, cy),

                        False
                    ) >= 0
                )

                if inside:

                    det["risk"] = (
                        zone["risk"]
                    )

                    det["danger_zone"] = True

                    det["zone_name"] = (
                        zone["name"]
                    )

                    det["priority"] = 5

                    alerts.append({

                        "zone":
                            zone["name"],

                        "severity":
                            zone["risk"],

                        "center":
                            [cx, cy]
                    })

    return alerts

# =========================================================
# MAIN
# =========================================================

def process_frame(
    frame
) -> Dict[str, Any]:

    if frame is None:
        return _empty()

    try:

        result = run_safety_pipeline(
            frame
        )

        detections = result.get(
            "detections",
            []
        )

        # =============================================
        # CLASSIFY
        # =============================================

        detections = [

            classify_detection(det)

            for det in detections
        ]

        # =============================================
        # INTRUSION
        # =============================================

        intrusion_alerts = analyze_intrusions(
            detections
        )

        # =============================================
        # ANALYTICS
        # =============================================

        analytics = calculate_operational_risk(
            detections
        )

        # =============================================
        # VISUALS
        # =============================================

        draw_danger_zones(
            frame,
            intrusion_alerts
        )

        draw_cinematic(
            frame,
            detections
        )

        draw_trajectories(
            frame,
            detections
        )

        draw_hud(
            frame,
            detections,
            intrusion_alerts,
            analytics
        )

        # =============================================
        # FINAL
        # =============================================

        return {

            **result,

            "detections":
                detections,

            "zones":
                DANGER_ZONES,

            "intrusions":
                intrusion_alerts,

            "analytics":
                analytics,

            "ai_metadata":
                build_ai_metadata()
        }

    except Exception:

        print("\n" + "=" * 80)
        print("[DETECTION ERROR] process_frame() failed:")
        traceback.print_exc()
        print("=" * 80 + "\n")

        return _empty()

# =========================================================
# TRAJECTORIES
# =========================================================

def draw_trajectories(
    frame,
    detections
):

    for det in detections:

        trajectory = det.get(
            "trajectory",
            []
        )

        if len(trajectory) < 2:
            continue

        risk = str(
            det.get(
                "risk",
                "LOW"
            )
        ).lower()

        color = {

            "critical":
                (0,0,255),

            "high":
                (0,80,255),

            "medium":
                (0,215,255),

            "low":
                (0,255,120)

        }.get(
            risk,
            (0,255,120)
        )

        for i in range(
            1,
            len(trajectory)
        ):

            cv2.line(

                frame,

                trajectory[i - 1],

                trajectory[i],

                color,

                2
            )

# =========================================================
# ZONES
# =========================================================

def draw_danger_zones(
    frame,
    intrusion_alerts
):

    pulse = abs(
        math.sin(
            time.time() * 3
        )
    )

    for zone in DANGER_ZONES:

        polygon = zone[
            "polygon"
        ]

        intrusion = any(

            a["zone"] == zone["name"]

            for a in intrusion_alerts
        )

        color = (

            (0,0,255)

            if intrusion

            else (255,120,0)
        )

        overlay = frame.copy()

        pts = np.array(
            polygon,
            np.int32
        )

        cv2.fillPoly(
            overlay,
            [pts],
            color
        )

        alpha = (
            0.22
            if intrusion
            else 0.08
        )

        frame[:] = cv2.addWeighted(
            overlay,
            alpha,
            frame,
            1 - alpha,
            0
        )

        cv2.polylines(
            frame,
            [pts],
            True,
            color,
            3
        )

# =========================================================
# BOXES
# =========================================================

def draw_cinematic(
    frame,
    detections
):

    for det in detections:

        bbox = det.get(
            "bbox",
            []
        )

        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(
            int,
            bbox
        )

        label = det.get(
            "class_name",
            "Object"
        )

        confidence = float(
            det.get(
                "confidence",
                0
            )
        )

        risk = str(
            det.get(
                "risk",
                "LOW"
            )
        ).lower()

        color = {

            "critical":
                (0,0,255),

            "high":
                (0,80,255),

            "medium":
                (0,215,255),

            "low":
                (0,255,120)

        }.get(
            risk,
            (0,255,120)
        )

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (x1,y1),
            (x2,y2),
            color,
            8
        )

        frame[:] = cv2.addWeighted(
            overlay,
            0.14,
            frame,
            0.86,
            0
        )

        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            color,
            2
        )

        text = (
            f"{label} "
            f"{confidence:.2f}"
        )

        cv2.rectangle(
            frame,
            (x1, y1 - 34),
            (x1 + 240, y1),
            color,
            -1
        )

        cv2.putText(

            frame,

            text,

            (x1 + 8, y1 - 10),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255,255,255),

            2
        )

# =========================================================
# HUD
# =========================================================

def draw_hud(

    frame,
    detections,
    intrusion_alerts,
    analytics
):

    h, w = frame.shape[:2]

    overlay = frame.copy()

    cv2.rectangle(

        overlay,

        (0,0),

        (w,52),

        (10,15,25),

        -1
    )

    frame[:] = cv2.addWeighted(
        overlay,
        0.6,
        frame,
        0.4,
        0
    )

    cv2.putText(

        frame,

        "InfraGuard Enterprise AI Surveillance",

        (18, 32),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.72,

        (255,255,255),

        2
    )

    cv2.putText(

        frame,

        f"Objects: {len(detections)}",

        (w - 260, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (0,255,120),

        2
    )

    cv2.putText(

        frame,

        f"Risk: {analytics['overall_risk']}",

        (w - 460, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (0,80,255),

        2
    )

    if intrusion_alerts:

        cv2.putText(

            frame,

            "DANGER ZONE INTRUSION",

            (w//2 - 180, 32),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.82,

            (0,0,255),

            3
        )

    timestamp = time.strftime(
        "%d-%m-%Y %H:%M:%S"
    )

    cv2.putText(

        frame,

        timestamp,

        (18, h - 18),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.56,

        (255,255,255),

        2
    )

# =========================================================
# EMPTY
# =========================================================

def _empty():

    return {

        "risk":
            "LOW",

        "detections": [],

        "zones": DANGER_ZONES,

        "intrusions": [],

        "analytics": {},

        "ai_metadata": {}
    }