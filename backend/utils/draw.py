"""
InfraGuard Enterprise AI — draw.py
Visualization layer for safety detection pipeline.
"""

import cv2
import math
import time
import numpy as np
from datetime import datetime


# ─────────────────────────────────────────────
#  CONFIGURATION CONSTANTS
# ─────────────────────────────────────────────

# Risk level → BGR color
RISK_COLORS = {
    "LOW":      (80,  200, 80),    # green
    "MEDIUM":   (0,   200, 255),   # yellow
    "HIGH":     (0,   120, 255),   # orange
    "CRITICAL": (0,   0,   220),   # red
}

# Per-class accent colors (BGR)
CLASS_COLORS = {
    "person":     (0,   200, 255),
    "helmet":     (0,   230, 80),
    "vest":       (0,   180, 255),
    "boots":      (60,  230, 180),
    "gloves":     (120, 220, 160),
    "crack":      (0,   210, 255),
    "excavator":  (200, 160, 255),
    "loader":     (180, 140, 255),
    "bulldozer":  (160, 120, 255),
}

# Box drawing
BOX_THICKNESS       = 2
CORNER_LENGTH       = 18       # length of corner accent lines
CORNER_THICKNESS    = 3
CORNER_RADIUS       = 6        # radius for rounded corner feel (decorative dots)

# Label panel
LABEL_FONT          = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE    = 0.48
LABEL_FONT_THICKNESS= 1
LABEL_PAD_X         = 6
LABEL_PAD_Y         = 5
LABEL_LINE_HEIGHT   = 17
LABEL_BG_ALPHA      = 0.72

# Header / footer
HEADER_HEIGHT       = 44
FOOTER_HEIGHT       = 30
PANEL_BG_ALPHA      = 0.65
HEADER_FONT_SCALE   = 0.52
STATS_FONT_SCALE    = 0.44

# Badge
BADGE_FONT_SCALE    = 0.38

# Trajectory
TRAJ_MAX_POINTS     = 30
TRAJ_FADE_STEPS     = 8

# PPE icons
PPE_TICK   = "\u2714"   # ✔  (rendered as text fallback)
PPE_CROSS  = "\u2716"   # ✖


# ─────────────────────────────────────────────
#  INTERNAL HELPERS
# ─────────────────────────────────────────────

def _color_for(det: dict) -> tuple:
    """Return BGR color based on risk level, falling back to class color."""
    risk = str(det.get("risk", "LOW")).upper()
    if risk in RISK_COLORS:
        return RISK_COLORS[risk]
    return CLASS_COLORS.get(str(det.get("class_name", "")).lower(), (200, 200, 200))


def _bbox(det: dict):
    """Normalise bbox from multiple possible input formats → (x1,y1,x2,y2) ints."""
    bbox = det.get("bbox")
    if not bbox:
        x = det.get("x", 0); y = det.get("y", 0)
        w = det.get("w", 0); h = det.get("h", 0)
        bbox = [x, y, x + w, y + h]
    if len(bbox) != 4:
        return None
    return tuple(map(int, bbox))


def _alpha_rect(frame, x1, y1, x2, y2, color, alpha):
    """Draw a filled rectangle with transparency."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _put_text(frame, text, x, y, scale=LABEL_FONT_SCALE,
              color=(255, 255, 255), thickness=LABEL_FONT_THICKNESS):
    cv2.putText(frame, text, (x, y), LABEL_FONT, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), LABEL_FONT, scale, color,     thickness,     cv2.LINE_AA)


def _text_size(text, scale=LABEL_FONT_SCALE, thickness=LABEL_FONT_THICKNESS):
    (w, h), baseline = cv2.getTextSize(text, LABEL_FONT, scale, thickness)
    return w, h, baseline


# ─────────────────────────────────────────────
#  ENTERPRISE BOUNDING BOX
# ─────────────────────────────────────────────

def draw_box(frame, x1, y1, x2, y2, color, thickness=BOX_THICKNESS):
    """
    Draw an enterprise-style bounding box:
    thin full-perimeter rectangle + bold corner accents.
    """
    # Dim full box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    cl = CORNER_LENGTH
    ct = CORNER_THICKNESS

    # Top-left
    cv2.line(frame, (x1, y1), (x1 + cl, y1), color, ct, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + cl), color, ct, cv2.LINE_AA)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - cl, y1), color, ct, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + cl), color, ct, cv2.LINE_AA)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + cl, y2), color, ct, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - cl), color, ct, cv2.LINE_AA)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - cl, y2), color, ct, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - cl), color, ct, cv2.LINE_AA)

    # Corner dots for a polished look
    for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        cv2.circle(frame, (cx, cy), CORNER_RADIUS, color, -1, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  LABEL PANEL
# ─────────────────────────────────────────────

def draw_label(frame, x1, y1, x2, det: dict, color: tuple):
    """
    Draw a multi-line label panel above (or inside) the bounding box.

    Lines rendered:
      • Class name  [+ tracking ID]   confidence%
      • PPE status  (for persons)
      • Crack severity                (for cracks)
      • Risk badge
      • Model badge
    """
    class_name  = str(det.get("class_name", det.get("label", "object")))
    risk        = str(det.get("risk", "LOW")).upper()
    confidence  = det.get("confidence")
    tracking_id = det.get("tracking_id", det.get("track_id"))
    ppe_status  = det.get("ppe_status", {})          # {"helmet": True, "vest": False, …}
    crack_sev   = det.get("crack_severity")           # "Low" / "Medium" / "High"
    model_name  = det.get("model", "InfraGuard")

    # ── Build label lines ──────────────────────────────────────────────
    lines = []

    # Line 1: class + ID + confidence
    line1 = class_name.capitalize()
    if tracking_id is not None:
        kind = "Worker" if class_name.lower() == "person" else class_name.capitalize()
        line1 = f"{kind} #{tracking_id}"
    if confidence is not None:
        line1 += f"  {float(confidence) * 100:.0f}%"
    lines.append((line1, (255, 255, 255)))

    # Line 2: PPE status (only for persons)
    if class_name.lower() == "person" and ppe_status:
        ppe_parts = []
        for item, ok in ppe_status.items():
            sym = "+" if ok else "-"          # ASCII-safe; cv2 can't render Unicode
            ppe_parts.append(f"{sym}{item[:3].capitalize()}")
        lines.append(("  ".join(ppe_parts), (180, 255, 180) if all(ppe_status.values()) else (80, 80, 255)))

    # Line 3: crack severity
    if class_name.lower() == "crack" and crack_sev:
        sev_color = RISK_COLORS.get(crack_sev.upper(), (0, 215, 255))
        lines.append((f"Severity: {crack_sev}", sev_color))

    # Line 4: risk tag
    risk_color = RISK_COLORS.get(risk, (200, 200, 200))
    lines.append((f"[{risk}]", risk_color))

    # Line 5: model badge
    lines.append((model_name, (180, 180, 180)))

    # ── Measure panel ─────────────────────────────────────────────────
    max_w = max(_text_size(t, LABEL_FONT_SCALE)[0] for t, _ in lines)
    panel_w = max_w + LABEL_PAD_X * 2
    panel_h = len(lines) * LABEL_LINE_HEIGHT + LABEL_PAD_Y * 2

    px1 = x1
    py2 = y1 - 2
    py1 = py2 - panel_h

    # Keep panel inside frame top
    if py1 < 0:
        py1 = y1 + 2
        py2 = py1 + panel_h

    px2 = px1 + panel_w

    # ── Draw panel background ──────────────────────────────────────────
    _alpha_rect(frame, px1, py1, px2, py2, (15, 15, 15), LABEL_BG_ALPHA)
    # Thin colored top border
    cv2.line(frame, (px1, py1), (px2, py1), color, 2, cv2.LINE_AA)

    # ── Render lines ───────────────────────────────────────────────────
    for i, (text, tcol) in enumerate(lines):
        tx = px1 + LABEL_PAD_X
        ty = py1 + LABEL_PAD_Y + (i + 1) * LABEL_LINE_HEIGHT - 3
        _put_text(frame, text, tx, ty, LABEL_FONT_SCALE, tcol)


# ─────────────────────────────────────────────
#  HEADER  (enterprise top bar)
# ─────────────────────────────────────────────

def draw_header(frame, camera_id=0, fps=0.0, pipeline_status="ONLINE",
                infraguard_loaded=True, crack_loaded=True):
    """
    Draw a translucent header bar at the top of the frame.

    Shows:  InfraGuard Enterprise AI | Camera N | FPS | Time | Pipeline Status
    """
    h_frame, w_frame = frame.shape[:2]
    _alpha_rect(frame, 0, 0, w_frame, HEADER_HEIGHT, (10, 10, 10), PANEL_BG_ALPHA)
    cv2.line(frame, (0, HEADER_HEIGHT), (w_frame, HEADER_HEIGHT), (60, 60, 60), 1)

    now = datetime.now().strftime("%H:%M:%S")

    left_text  = "InfraGuard Enterprise AI"
    center_text = f"Camera {camera_id}   |   {fps:.1f} FPS   |   {now}"

    # AI status indicators
    ig_col  = (0, 230, 80)  if infraguard_loaded else (0, 0, 220)
    cr_col  = (0, 230, 80)  if crack_loaded       else (0, 0, 220)
    pp_col  = (0, 230, 80)  if pipeline_status == "ONLINE" else (0, 80, 255)

    ig_tag  = f"InfraGuard {'OK' if infraguard_loaded else 'ERR'}"
    cr_tag  = f"Crack {'OK' if crack_loaded else 'ERR'}"
    pp_tag  = f"AI {pipeline_status}"

    # Left: brand
    _put_text(frame, left_text, 10, 28, HEADER_FONT_SCALE, (0, 200, 255), 1)

    # Center: camera / fps / time
    cw = _text_size(center_text, HEADER_FONT_SCALE)[0]
    _put_text(frame, center_text, (w_frame - cw) // 2, 28, HEADER_FONT_SCALE, (220, 220, 220), 1)

    # Right: status badges
    right_x = w_frame - 10
    for tag, tcol in reversed([(pp_tag, pp_col), (ig_tag, ig_col), (cr_tag, cr_col)]):
        tw = _text_size(tag, BADGE_FONT_SCALE)[0]
        right_x -= tw + 14
        bx1, by1, bx2, by2 = right_x - 4, 8, right_x + tw + 4, 36
        _alpha_rect(frame, bx1, by1, bx2, by2, (30, 30, 30), 0.8)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), tcol, 1, cv2.LINE_AA)
        _put_text(frame, tag, right_x, 28, BADGE_FONT_SCALE, tcol, 1)


# ─────────────────────────────────────────────
#  FOOTER  (stats bar)
# ─────────────────────────────────────────────

def draw_footer(frame, detections):
    """
    Draw a translucent footer bar at the bottom showing detection counts.
    """
    h_frame, w_frame = frame.shape[:2]
    fy = h_frame - FOOTER_HEIGHT

    _alpha_rect(frame, 0, fy, w_frame, h_frame, (10, 10, 10), PANEL_BG_ALPHA)
    cv2.line(frame, (0, fy), (w_frame, fy), (60, 60, 60), 1)

    persons   = sum(1 for d in detections if str(d.get("class_name","")).lower() == "person")
    equipment = sum(1 for d in detections if str(d.get("class_name","")).lower()
                    in ("excavator","loader","bulldozer"))
    cracks    = sum(1 for d in detections if str(d.get("class_name","")).lower() == "crack")
    alerts    = sum(1 for d in detections if str(d.get("risk","LOW")).upper()
                    in ("HIGH","CRITICAL"))

    stats = [
        (f"Persons : {persons}",   (0, 200, 255)),
        (f"Equipment : {equipment}",(200, 160, 255)),
        (f"Cracks : {cracks}",     (0, 210, 255)),
        (f"Alerts : {alerts}",     (0, 80, 255) if alerts else (120, 120, 120)),
    ]

    x = 12
    for text, col in stats:
        _put_text(frame, text, x, h_frame - 10, STATS_FONT_SCALE, col, 1)
        x += _text_size(text, STATS_FONT_SCALE)[0] + 28


# ─────────────────────────────────────────────
#  STATISTICS PANEL  (top-left below header)
# ─────────────────────────────────────────────

def draw_statistics(frame, detections, latency_ms=0.0):
    """
    Small stats block below the header (frame number / latency etc).
    Kept minimal to avoid clutter — extend as needed.
    """
    # Already covered in header/footer; this is a hook for per-frame extended stats.
    pass


# ─────────────────────────────────────────────
#  WARNING OVERLAY
# ─────────────────────────────────────────────

def draw_warning(frame, message: str, level: str = "HIGH"):
    """Flash a semi-transparent warning banner across the middle of the frame."""
    pulse = 0.5 + 0.5 * abs(math.sin(time.time() * 4))
    h_frame, w_frame = frame.shape[:2]
    by1 = h_frame // 2 - 22
    by2 = h_frame // 2 + 22
    col = RISK_COLORS.get(level.upper(), (0, 80, 255))
    _alpha_rect(frame, 0, by1, w_frame, by2, col, 0.25 + 0.15 * pulse)
    tw = _text_size(message, 0.7, 2)[0]
    _put_text(frame, message, (w_frame - tw) // 2, h_frame // 2 + 7, 0.7, (255, 255, 255), 2)


# ─────────────────────────────────────────────
#  ZONE OVERLAY
# ─────────────────────────────────────────────

def draw_zone(frame, zone: dict):
    """
    Draw a single danger zone polygon with animated pulsing fill,
    a solid border, and an optional zone label.
    """
    polygon = zone.get("polygon")
    if not polygon:
        return

    pulse = abs(math.sin(time.time() * 3))
    pts   = np.array(polygon, dtype="int32")

    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (0, 0, 200))
    alpha = 0.12 + pulse * 0.10
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.polylines(frame, [pts], True, (0, 0, 220), 2, cv2.LINE_AA)

    label = zone.get("label", "Danger Zone")
    cx = int(np.mean([p[0] for p in polygon]))
    cy = int(np.mean([p[1] for p in polygon]))
    _put_text(frame, label, cx - 40, cy, BADGE_FONT_SCALE, (0, 0, 220), 1)


# ─────────────────────────────────────────────
#  PUBLIC DRAWING FUNCTIONS
# ─────────────────────────────────────────────

def draw_detections(frame, detections: list) -> np.ndarray:
    """
    Draw all detection bounding boxes + enterprise label panels.
    Backward-compatible with old x/y/w/h dicts and new bbox dicts.
    """
    for det in detections:
        coords = _bbox(det)
        if coords is None:
            continue
        x1, y1, x2, y2 = coords
        color = _color_for(det)
        draw_box(frame, x1, y1, x2, y2, color)
        draw_label(frame, x1, y1, x2, det, color)
    return frame


def draw_trajectories(frame, detections: list) -> np.ndarray:
    """
    Draw fading movement trails from trajectory history.
    """
    for det in detections:
        points = det.get("trajectory", [])
        if len(points) < 2:
            continue

        color  = _color_for(det)
        n      = len(points)
        steps  = min(TRAJ_FADE_STEPS, n - 1)

        for i in range(1, n):
            alpha_factor = i / n
            faded = tuple(int(c * alpha_factor) for c in color)
            thickness = max(1, int(2 * alpha_factor))
            cv2.line(frame, tuple(points[i - 1]), tuple(points[i]),
                     faded, thickness, cv2.LINE_AA)

        # Arrow at last point indicating direction
        if n >= 2:
            p1 = np.array(points[-2], dtype=float)
            p2 = np.array(points[-1], dtype=float)
            direction = p2 - p1
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm * 10
                tip = tuple(points[-1])
                tail = tuple((p2 - direction).astype(int))
                cv2.arrowedLine(frame, tail, tip, color, 2, cv2.LINE_AA, tipLength=0.5)

    return frame


def draw_danger_zones(frame, zones: list) -> np.ndarray:
    """Draw all danger zone overlays."""
    for zone in zones:
        draw_zone(frame, zone)
    return frame


def draw_frame(
    frame,
    detections,
    zones=None,
    camera_id=0,
    fps=0.0,
    pipeline_status="ONLINE",
    infraguard_loaded=True,
    crack_loaded=True,
    warnings=None,
) -> np.ndarray:
    """
    Complete enterprise renderer — entry point for detection_service.py.

    Call order:
      1. Danger zones (background)
      2. Trajectories
      3. Detection boxes + labels
      4. Header bar
      5. Footer stats bar
      6. Warning banners (if any)
    """
    if zones:
        draw_danger_zones(frame, zones)

    draw_trajectories(frame, detections)
    draw_detections(frame, detections)

    draw_header(
        frame,
        camera_id=camera_id,
        fps=fps,
        pipeline_status=pipeline_status,
        infraguard_loaded=infraguard_loaded,
        crack_loaded=crack_loaded,
    )

    draw_footer(frame, detections)

    if warnings:
        for msg, level in warnings:
            draw_warning(frame, msg, level)

    return frame