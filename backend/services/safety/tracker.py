import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATE_NEW    = "NEW"
STATE_ACTIVE = "ACTIVE"
STATE_LOST   = "LOST"

EQUIPMENT_LABELS = {
    "excavator", "bulldozer", "loader", "roller",
    "grader", "dump_truck", "mobile_crane",
}

_ALPHA = 0.5


# ---------------------------------------------------------------------------
# TrackerConfig  (Priority 14)
# ---------------------------------------------------------------------------

@dataclass
class TrackerConfig:
    """Centralised configuration — tune here instead of hunting kwargs."""
    max_distance:         float = 90.0
    max_missing:          float = 20.0
    history_size:         int   = 35
    confirm_frames:       int   = 3
    reid_distance:        float = 120.0
    idle_speed_threshold: float = 5.0
    debug:                bool  = False   # Priority 11


# ---------------------------------------------------------------------------
# TrackerStats  (Priority 13)
# ---------------------------------------------------------------------------

@dataclass
class TrackerStats:
    """Tracker-level aggregate metrics."""
    tracks_created:   int   = 0
    tracks_removed:   int   = 0
    total_frames:     int   = 0
    lifetime_sum:     float = 0.0   # seconds, for avg lifetime
    speed_sum:        float = 0.0
    speed_count:      int   = 0

    @property
    def avg_lifetime(self) -> float:
        removed = self.tracks_removed or 1
        return self.lifetime_sum / removed

    @property
    def avg_speed(self) -> float:
        count = self.speed_count or 1
        return self.speed_sum / count


# ---------------------------------------------------------------------------
# EnterpriseTracker
# ---------------------------------------------------------------------------

class EnterpriseTracker:
    """
    Multi-object tracker with:
      - Hungarian assignment + IoU cost
      - Class-aware matching  (Priority 1)
      - Detection validation  (Priority 2)
      - Track lifecycle logging (Priority 3)
      - Per-frame summary      (Priority 4)
      - Extended confidence metrics (Priority 5)
      - Lost-track diagnostics (Priority 6)
      - Multi-cue ReID         (Priority 7)
      - Track health label     (Priority 8)
      - Equipment analytics    (Priority 9)
      - Per-frame telemetry    (Priority 10)
      - Debug mode             (Priority 11)
      - Assignment diagnostics (Priority 12)
      - Tracker-level stats    (Priority 13)
      - TrackerConfig object   (Priority 14)
      - Exception logging      (Priority 15)
    """

    def __init__(self, config: Optional[TrackerConfig] = None, **kwargs):
        # Accept either a TrackerConfig or legacy keyword arguments
        if config is None:
            config = TrackerConfig(**{
                k: v for k, v in kwargs.items()
                if k in TrackerConfig.__dataclass_fields__
            })
        self.cfg = config

        # Convenience aliases (keeps internal code readable)
        self.max_distance         = config.max_distance
        self.max_missing          = config.max_missing
        self.history_size         = config.history_size
        self.confirm_frames       = config.confirm_frames
        self.reid_distance        = config.reid_distance
        self.idle_speed_threshold = config.idle_speed_threshold
        self.debug                = config.debug

        self.tracks: dict      = {}
        self.stats             = TrackerStats()
        self._lock             = threading.Lock()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def update(self, detections: list) -> list:
        with self._lock:
            try:
                return self._update_locked(detections)
            except Exception as exc:  # Priority 15
                logger.error("Tracker.update() failed: %s", exc, exc_info=True)
                return detections

    def get_active_tracks(self) -> list:
        with self._lock:
            return [
                dict(t) for t in self.tracks.values()
                if t.get("state") == STATE_ACTIVE
            ]

    def get_all_tracks(self) -> list:
        with self._lock:
            return [dict(t) for t in self.tracks.values()]

    def get_stats(self) -> dict:
        """Return a snapshot of tracker-level statistics."""
        with self._lock:
            active  = sum(1 for t in self.tracks.values() if t["state"] == STATE_ACTIVE)
            lost    = sum(1 for t in self.tracks.values() if t["state"] == STATE_LOST)
            new     = sum(1 for t in self.tracks.values() if t["state"] == STATE_NEW)
            return {
                "tracks_created":  self.stats.tracks_created,
                "tracks_removed":  self.stats.tracks_removed,
                "total_frames":    self.stats.total_frames,
                "avg_lifetime_s":  round(self.stats.avg_lifetime, 2),
                "avg_speed":       round(self.stats.avg_speed, 2),
                "active":          active,
                "lost":            lost,
                "new":             new,
            }

    def reset(self):
        with self._lock:
            self.tracks.clear()
            self.stats = TrackerStats()

    # -----------------------------------------------------------------------
    # Core update (locked)
    # -----------------------------------------------------------------------

    def _update_locked(self, detections: list) -> list:
        now = time.time()
        t0  = time.perf_counter()
        self.stats.total_frames += 1

        # ── Detection validation  (Priority 2) ──────────────────────────────
        valid_indices: list[int]  = []
        det_centers:  list[tuple] = []
        det_bboxes:   list[list]  = []
        det_labels:   list[str]   = []
        det_confs:    list[float] = []
        rejected = 0

        for i, det in enumerate(detections):
            ok, reason = self._validate_detection(det)
            if not ok:
                rejected += 1
                if self.debug:
                    logger.debug("Detection %d rejected: %s", i, reason)
                continue
            bbox  = det["bbox"]
            label = str(det.get("label", det.get("class", ""))).lower()
            conf  = float(det.get("confidence", det.get("score", 1.0)))
            valid_indices.append(i)
            det_centers.append(self._center(bbox))
            det_bboxes.append(bbox)
            det_labels.append(label)
            det_confs.append(conf)

        # ── Separate active/lost tracks ──────────────────────────────────────
        active_ids = [
            tid for tid, t in self.tracks.items()
            if t["state"] in (STATE_ACTIVE, STATE_NEW)
        ]
        lost_ids = [
            tid for tid, t in self.tracks.items()
            if t["state"] == STATE_LOST
        ]

        # ── Primary assignment  ──────────────────────────────────────────────
        t_assign = time.perf_counter()
        matches  = self._assign(det_centers, det_bboxes, det_labels, active_ids)
        assign_ms = (time.perf_counter() - t_assign) * 1000

        # ── ReID against LOST  (Priority 7) ─────────────────────────────────
        unmatched_local = [
            li for li in range(len(valid_indices))
            if li not in matches
        ]
        t_reid = time.perf_counter()
        if unmatched_local and lost_ids:
            reid_matches = self._reid(
                unmatched_local, det_centers, det_bboxes, det_labels, det_confs, lost_ids
            )
            matches.update(reid_matches)
        reid_ms = (time.perf_counter() - t_reid) * 1000

        assigned_track_ids = set(matches.values())

        # ── Apply matches ────────────────────────────────────────────────────
        updated_tracks: dict = {}
        new_count = matched_count = 0

        for local_i, original_i in enumerate(valid_indices):
            det    = detections[original_i]
            bbox   = det_bboxes[local_i]
            center = det_centers[local_i]
            conf   = det_confs[local_i]
            label  = det_labels[local_i]

            matched_id = matches.get(local_i)

            try:
                if matched_id is not None:
                    track   = self.tracks[matched_id]
                    updated = self._update_track(track, center, bbox, conf, label, now)
                    updated_tracks[matched_id] = updated
                    self._annotate_detection(det, updated)
                    matched_count += 1
                else:
                    new_id  = self._new_id()
                    new_trk = self._create_track(new_id, center, bbox, conf, label, now)
                    updated_tracks[new_id] = new_trk
                    self._annotate_detection(det, new_trk)
                    new_count += 1
                    self.stats.tracks_created += 1
                    if self.debug:
                        logger.debug("Track %s CREATED  label=%s", new_id, label)
            except Exception as exc:  # Priority 15
                logger.error(
                    "Error processing detection local_i=%d matched_id=%s: %s",
                    local_i, matched_id, exc, exc_info=True,
                )

        # ── Carry forward unmatched tracks  ─────────────────────────────────
        lost_count = removed_count = 0

        for track_id, track in self.tracks.items():
            if track_id in assigned_track_ids:
                continue

            last_seen = track.get("last_seen", now)
            missing   = now - last_seen

            if missing >= self.max_missing:
                # ── Lost-track diagnostics  (Priority 6) ────────────────────
                lifetime = track.get("age", 0.0)
                self.stats.tracks_removed  += 1
                self.stats.lifetime_sum    += lifetime
                removed_count += 1
                logger.info(
                    "Track %s REMOVED  reason=timeout  missing_frames=~%d  "
                    "last_pos=%s  label=%s  lifetime=%.1fs",
                    track_id,
                    round(missing * 30),   # approx frames at 30 fps
                    track.get("center"),
                    track.get("label"),
                    lifetime,
                )
                continue

            # ── Lifecycle logging  (Priority 3) ─────────────────────────────
            t_copy = dict(track)
            if t_copy["state"] != STATE_LOST:
                logger.info(
                    "Track %s  %s → LOST  label=%s",
                    track_id, t_copy["state"], t_copy.get("label"),
                )
                t_copy["state"] = STATE_LOST
                lost_count += 1
            updated_tracks[track_id] = t_copy

        self.tracks = updated_tracks
        total_ms = (time.perf_counter() - t0) * 1000

        # ── Per-frame telemetry  (Priority 10) ──────────────────────────────
        telemetry = {
            "assign_ms":  round(assign_ms, 2),
            "reid_ms":    round(reid_ms, 2),
            "total_ms":   round(total_ms, 2),
        }

        # ── Per-frame summary  (Priority 4) ─────────────────────────────────
        active_now = sum(1 for t in self.tracks.values() if t["state"] == STATE_ACTIVE)
        logger.info(
            "Frame %d | raw=%d  valid=%d  rejected=%d  matched=%d  "
            "new=%d  lost=%d  removed=%d  active=%d | "
            "assign=%.1fms  reid=%.1fms  total=%.1fms",
            self.stats.total_frames,
            len(detections), len(valid_indices), rejected,
            matched_count, new_count, lost_count, removed_count,
            active_now,
            assign_ms, reid_ms, total_ms,
        )

        if self.debug:
            logger.debug("Telemetry: %s", telemetry)

        return detections

    # -----------------------------------------------------------------------
    # Validation  (Priority 2)
    # -----------------------------------------------------------------------

    @staticmethod
    def _validate_detection(det: dict):
        """Returns (True, None) or (False, reason_str)."""
        bbox = det.get("bbox", [])
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False, f"bbox length {len(bbox)} != 4"
        x1, y1, x2, y2 = bbox
        for v in (x1, y1, x2, y2):
            if not isinstance(v, (int, float)):
                return False, f"non-numeric bbox coord: {v}"
        if x2 <= x1 or y2 <= y1:
            return False, f"degenerate bbox {bbox}"
        conf = det.get("confidence", det.get("score"))
        if conf is not None:
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                return False, f"non-numeric confidence: {conf}"
            if not (0.0 <= conf <= 1.0):
                return False, f"confidence {conf} out of [0, 1]"
        label = det.get("label", det.get("class", ""))
        if label is None:
            return False, "missing label/class"
        return True, None

    # -----------------------------------------------------------------------
    # Assignment  (Priority 1 class-aware, Priority 12 diagnostics)
    # -----------------------------------------------------------------------

    def _assign(
        self,
        det_centers: list,
        det_bboxes:  list,
        det_labels:  list,
        track_ids:   list,
    ) -> dict:
        if not det_centers or not track_ids:
            return {}

        n_det = len(det_centers)
        n_trk = len(track_ids)
        cost  = np.full((n_det, n_trk), fill_value=1e6, dtype=np.float64)

        for i, (center, bbox, label) in enumerate(zip(det_centers, det_bboxes, det_labels)):
            for j, tid in enumerate(track_ids):
                trk = self.tracks[tid]

                # ── Class-aware gate  (Priority 1) ──────────────────────────
                if label and trk.get("label") and label != trk["label"]:
                    if self.debug:
                        logger.debug(
                            "Assignment det=%d → track=%s REJECTED class mismatch "
                            "(%s vs %s)", i, tid, label, trk["label"]
                        )
                    continue   # leave cost at 1e6 → Hungarian will never pick it

                dist   = self._distance(center, trk["center"])
                norm_d = min(dist / max(self.max_distance, 1.0), 1.0)
                iou    = self._iou(bbox, trk["bbox"])
                cost[i, j] = _ALPHA * norm_d + (1.0 - _ALPHA) * (1.0 - iou)

        det_idx, trk_idx = linear_sum_assignment(cost)

        matches: dict = {}
        for d_i, t_i in zip(det_idx, trk_idx):
            if cost[d_i, t_i] >= 1e5:   # class-mismatch sentinel
                continue
            trk  = self.tracks[track_ids[t_i]]
            dist = self._distance(det_centers[d_i], trk["center"])
            if dist < self.max_distance:
                matches[d_i] = track_ids[t_i]
            else:
                if self.debug:  # Priority 12
                    logger.debug(
                        "Assignment det=%d → track=%s REJECTED distance=%.1f > %.1f",
                        d_i, track_ids[t_i], dist, self.max_distance,
                    )

        return matches

    # -----------------------------------------------------------------------
    # Re-identification  (Priority 7 — multi-cue)
    # -----------------------------------------------------------------------

    def _reid(
        self,
        unmatched_local: list,
        det_centers:     list,
        det_bboxes:      list,
        det_labels:      list,
        det_confs:       list,
        lost_ids:        list,
    ) -> dict:
        matches:   dict = {}
        used_lost: set  = set()

        for li in unmatched_local:
            center = det_centers[li]
            bbox   = det_bboxes[li]
            label  = det_labels[li]
            conf   = det_confs[li]

            best_score = float("inf")
            best_id    = None

            for tid in lost_ids:
                if tid in used_lost:
                    continue
                trk = self.tracks[tid]

                # class gate
                if label and trk.get("label") and label != trk["label"]:
                    continue

                dist = self._distance(center, trk["center"])
                if dist >= self.reid_distance:
                    continue

                iou    = self._iou(bbox, trk["bbox"])
                conf_d = abs(conf - trk.get("avg_confidence", conf))
                # multi-cue score: lower is better
                score  = (dist / max(self.reid_distance, 1.0)) * 0.5 \
                       + (1.0 - iou) * 0.3 \
                       + conf_d * 0.2

                if score < best_score:
                    best_score = score
                    best_id    = tid

            if best_id is not None:
                matches[li] = best_id
                used_lost.add(best_id)
                if self.debug:
                    logger.debug(
                        "ReID det_local=%d → track=%s  score=%.3f  label=%s",
                        li, best_id, best_score, label,
                    )

        return matches

    # -----------------------------------------------------------------------
    # Track creation / update
    # -----------------------------------------------------------------------

    def _create_track(
        self,
        track_id: str,
        center:   tuple,
        bbox:     list,
        conf:     float,
        label:    str,
        now:      float,
    ) -> dict:
        is_equipment = label in EQUIPMENT_LABELS
        return {
            "id":               track_id,
            "state":            STATE_NEW,
            "label":            label,
            "is_equipment":     is_equipment,
            "center":           center,
            "bbox":             bbox,
            "trajectory":       [center],
            "first_seen":       now,
            "last_seen":        now,
            "frames":           1,
            "confirm_frames":   1,
            "age":              0.0,
            "velocity":         None,
            "direction":        None,
            "total_distance":   0.0,
            "avg_speed":        0.0,
            "max_speed":        0.0,
            "idle_time":        0.0,
            # Priority 5 — extended confidence metrics
            "confidences":      [conf],
            "avg_confidence":   conf,
            "max_confidence":   conf,
            "min_confidence":   conf,
            "cur_confidence":   conf,
            "zone":             None,
            # Priority 8 — track health
            "health":           "stable",
            "lost_count":       0,
            # Priority 9 — equipment analytics
            "operating_time":   0.0,
        }

    def _update_track(
        self,
        track:  dict,
        center: tuple,
        bbox:   list,
        conf:   float,
        label:  str,
        now:    float,
    ) -> dict:
        t = dict(track)

        # trajectory
        traj = t["trajectory"][:]
        traj.append(center)
        t["trajectory"] = traj[-self.history_size:]

        # timing
        dt = max(now - t.get("last_seen", now), 0.0)
        t["last_seen"] = now
        t["age"]       = now - t.get("first_seen", now)
        t["frames"]    = t.get("frames", 0) + 1

        # ── State machine + lifecycle logging  (Priority 3) ─────────────────
        old_state = t["state"]
        cf = t.get("confirm_frames", 0) + 1
        t["confirm_frames"] = cf
        if old_state == STATE_NEW and cf >= self.confirm_frames:
            t["state"] = STATE_ACTIVE
        elif old_state == STATE_LOST:
            t["state"]      = STATE_ACTIVE
            t["lost_count"] = t.get("lost_count", 0) + 1

        if t["state"] != old_state:
            logger.info(
                "Track %s  %s → %s  label=%s  frames=%d",
                t["id"], old_state, t["state"], t.get("label"), t["frames"],
            )

        # velocity / direction
        prev_center = t.get("center")
        velocity, direction = self._velocity_and_direction(prev_center, dt, center)
        t["velocity"]  = velocity
        t["direction"] = direction

        # odometry
        if prev_center is not None:
            seg = self._distance(prev_center, center)
            t["total_distance"] = t.get("total_distance", 0.0) + seg

        # speed
        speed = math.hypot(*velocity) if velocity is not None else 0.0
        frames = t["frames"]
        t["avg_speed"] = (t.get("avg_speed", 0.0) * (frames - 1) + speed) / frames
        t["max_speed"] = max(t.get("max_speed", 0.0), speed)

        # idle / operating time
        if dt > 0:
            if speed < self.idle_speed_threshold:
                t["idle_time"] = t.get("idle_time", 0.0) + dt
            else:
                t["operating_time"] = t.get("operating_time", 0.0) + dt  # Priority 9

        # ── Extended confidence  (Priority 5) ───────────────────────────────
        confs = t.get("confidences", [])
        confs.append(conf)
        confs = confs[-self.history_size:]
        t["confidences"]    = confs
        t["avg_confidence"] = float(np.mean(confs))
        t["max_confidence"] = float(max(confs))
        t["min_confidence"] = float(min(confs))
        t["cur_confidence"] = conf
        self.stats.speed_sum   += speed
        self.stats.speed_count += 1

        # ── Track health  (Priority 8) ───────────────────────────────────────
        lost_count = t.get("lost_count", 0)
        if lost_count == 0:
            t["health"] = "stable"
        elif lost_count == 1:
            t["health"] = "recently_recovered"
        elif lost_count <= 3:
            t["health"] = "unstable"
        else:
            t["health"] = "lost_frequently"

        # geometry
        t["center"]       = center
        t["bbox"]         = bbox
        t["label"]        = label or t.get("label", "")
        t["is_equipment"] = t["label"] in EQUIPMENT_LABELS

        return t

    # -----------------------------------------------------------------------
    # Detection annotation
    # -----------------------------------------------------------------------

    @staticmethod
    def _annotate_detection(det: dict, track: dict):
        det["track_id"]       = track["id"]
        det["track_state"]    = track["state"]
        det["track_health"]   = track.get("health", "stable")    # Priority 8
        det["trajectory"]     = track["trajectory"]
        det["frames_tracked"] = track["frames"]
        det["age"]            = track["age"]
        det["velocity"]       = track["velocity"]
        det["direction"]      = track["direction"]
        det["total_distance"] = track["total_distance"]
        det["avg_speed"]      = track["avg_speed"]
        det["max_speed"]      = track["max_speed"]
        det["idle_time"]      = track["idle_time"]
        det["operating_time"] = track.get("operating_time", 0.0)  # Priority 9
        det["avg_confidence"] = track["avg_confidence"]
        det["max_confidence"] = track["max_confidence"]
        det["min_confidence"] = track.get("min_confidence", track["avg_confidence"])  # Priority 5
        det["cur_confidence"] = track.get("cur_confidence", track["avg_confidence"])  # Priority 5
        det["last_seen"]      = track["last_seen"]
        det["zone"]           = track.get("zone")
        det["is_equipment"]   = track.get("is_equipment", False)

    # -----------------------------------------------------------------------
    # Geometry helpers  (unchanged — already solid)
    # -----------------------------------------------------------------------

    @staticmethod
    def _center(bbox) -> tuple:
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    @staticmethod
    def _distance(c1: tuple, c2: tuple) -> float:
        return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

    @staticmethod
    def _iou(bbox_a: list, bbox_b: list) -> float:
        """Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2);  iy2 = min(ay2, by2)
        iw  = max(0, ix2 - ix1)
        ih  = max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union  = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _velocity_and_direction(prev_center, dt: float, center: tuple):
        if prev_center is None or dt <= 0:
            return None, None
        vx = (center[0] - prev_center[0]) / dt
        vy = (center[1] - prev_center[1]) / dt
        return (vx, vy), math.degrees(math.atan2(vy, vx))

    @staticmethod
    def _new_id() -> str:
        return str(uuid.uuid4())[:8]