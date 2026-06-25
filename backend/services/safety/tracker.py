
import math
import threading
import time
import uuid

import numpy as np
from scipy.optimize import linear_sum_assignment


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
# EnterpriseTracker
# ---------------------------------------------------------------------------

class EnterpriseTracker:
    

    def __init__(
        self,
        max_distance: float = 90.0,
        max_missing: float  = 20.0,
        history_size: int   = 35,
        confirm_frames: int = 3,
        reid_distance: float = 120.0,
        idle_speed_threshold: float = 5.0,
    ):
        self.max_distance         = max_distance
        self.max_missing          = max_missing
        self.history_size         = history_size
        self.confirm_frames       = confirm_frames
        self.reid_distance        = reid_distance
        self.idle_speed_threshold = idle_speed_threshold

        # track_id → track dict
        self.tracks: dict = {}
        self._lock = threading.Lock()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def update(self, detections: list) -> list:
       
        with self._lock:
            return self._update_locked(detections)

    def get_active_tracks(self) -> list:
        """Return a shallow copy of all ACTIVE track dicts."""
        with self._lock:
            return [
                dict(t) for t in self.tracks.values()
                if t.get("state") == STATE_ACTIVE
            ]

    def get_all_tracks(self) -> list:
        """Return a shallow copy of every track dict."""
        with self._lock:
            return [dict(t) for t in self.tracks.values()]

    def reset(self):
        """Drop all tracks (e.g. on stream restart)."""
        with self._lock:
            self.tracks.clear()

   
    def _update_locked(self, detections: list) -> list:
        now = time.time()

        # -- collect valid detections ----------------------------------------
        valid_indices: list[int]   = []
        det_centers:  list[tuple]  = []
        det_bboxes:   list[list]   = []

        for i, det in enumerate(detections):
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue
            valid_indices.append(i)
            det_centers.append(self._center(bbox))
            det_bboxes.append(bbox)

        # -- separate ACTIVE/NEW and LOST tracks for assignment --------------
        active_ids = [
            tid for tid, t in self.tracks.items()
            if t["state"] in (STATE_ACTIVE, STATE_NEW)
        ]
        lost_ids = [
            tid for tid, t in self.tracks.items()
            if t["state"] == STATE_LOST
        ]

        # -- primary assignment against active tracks ------------------------
        matches = self._assign(det_centers, det_bboxes, active_ids)

        # -- re-identification against LOST tracks ---------------------------
        unmatched_local = [
            li for li in range(len(valid_indices))
            if li not in matches
        ]
        if unmatched_local and lost_ids:
            reid_matches = self._reid(
                unmatched_local, det_centers, lost_ids
            )
            matches.update(reid_matches)

        assigned_track_ids = set(matches.values())

        # -- apply matches ----------------------------------------------------
        updated_tracks: dict = {}

        for local_i, original_i in enumerate(valid_indices):
            det    = detections[original_i]
            bbox   = det_bboxes[local_i]
            center = det_centers[local_i]
            conf   = float(det.get("confidence", det.get("score", 1.0)))
            label  = str(det.get("label", det.get("class", ""))).lower()

            matched_id = matches.get(local_i)

            if matched_id is not None:
                # ---- update existing track ---------------------------------
                track   = self.tracks[matched_id]
                updated = self._update_track(
                    track, center, bbox, conf, label, now
                )
                updated_tracks[matched_id] = updated
                self._annotate_detection(det, updated)
            else:
                # ---- create new track --------------------------------------
                new_id  = self._new_id()
                new_trk = self._create_track(
                    new_id, center, bbox, conf, label, now
                )
                updated_tracks[new_id] = new_trk
                self._annotate_detection(det, new_trk)

        # -- carry forward unmatched tracks as LOST/pruned ------------------
        for track_id, track in self.tracks.items():
            if track_id in assigned_track_ids:
                continue

            last_seen = track.get("last_seen", now)
            missing   = now - last_seen

            if missing >= self.max_missing:
                # REMOVED — drop silently
                continue

            # Promote to LOST if not already
            t_copy         = dict(track)
            t_copy["state"] = STATE_LOST
            updated_tracks[track_id] = t_copy

        self.tracks = updated_tracks
        return detections

    # -----------------------------------------------------------------------
    # Assignment
    # -----------------------------------------------------------------------

    def _assign(
        self,
        det_centers: list,
        det_bboxes:  list,
        track_ids:   list,
    ) -> dict:
       
        if not det_centers or not track_ids:
            return {}

        n_det = len(det_centers)
        n_trk = len(track_ids)
        cost  = np.zeros((n_det, n_trk), dtype=np.float64)

        for i, (center, bbox) in enumerate(zip(det_centers, det_bboxes)):
            for j, tid in enumerate(track_ids):
                trk    = self.tracks[tid]
                dist   = self._distance(center, trk["center"])
                norm_d = min(dist / max(self.max_distance, 1.0), 1.0)

                iou    = self._iou(bbox, trk["bbox"])
                cost[i, j] = _ALPHA * norm_d + (1.0 - _ALPHA) * (1.0 - iou)

        det_idx, trk_idx = linear_sum_assignment(cost)

        matches: dict = {}
        for d_i, t_i in zip(det_idx, trk_idx):
            trk   = self.tracks[track_ids[t_i]]
            dist  = self._distance(det_centers[d_i], trk["center"])
            if dist < self.max_distance:
                matches[d_i] = track_ids[t_i]

        return matches

    def _reid(
        self,
        unmatched_local: list,
        det_centers:     list,
        lost_ids:        list,
    ) -> dict:
        
        matches: dict = {}
        used_lost: set = set()

        for li in unmatched_local:
            center = det_centers[li]
            best_d  = self.reid_distance
            best_id = None

            for tid in lost_ids:
                if tid in used_lost:
                    continue
                d = self._distance(center, self.tracks[tid]["center"])
                if d < best_d:
                    best_d  = d
                    best_id = tid

            if best_id is not None:
                matches[li] = best_id
                used_lost.add(best_id)

        return matches

    # -----------------------------------------------------------------------
    # Track creation / update helpers
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
            "id":             track_id,
            "state":          STATE_NEW,
            "label":          label,
            "is_equipment":   is_equipment,
            "center":         center,
            "bbox":           bbox,
            "trajectory":     [center],
            "first_seen":     now,
            "last_seen":      now,
            "frames":         1,
            "confirm_frames": 1,
            "age":            0.0,
            "velocity":       None,
            "direction":      None,
            "total_distance": 0.0,
            "avg_speed":      0.0,
            "max_speed":      0.0,
            "idle_time":      0.0,
            "confidences":    [conf],
            "avg_confidence": conf,
            "max_confidence": conf,
            "zone":           None,
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
        dt         = max(now - t.get("last_seen", now), 0.0)
        first_seen = t.get("first_seen", now)
        t["last_seen"] = now
        t["age"]       = now - first_seen
        t["frames"]    = t.get("frames", 0) + 1

        # state machine: NEW → ACTIVE after confirm_frames hits
        cf = t.get("confirm_frames", 0) + 1
        t["confirm_frames"] = cf
        if t["state"] == STATE_NEW and cf >= self.confirm_frames:
            t["state"] = STATE_ACTIVE
        elif t["state"] == STATE_LOST:
            # re-acquired
            t["state"] = STATE_ACTIVE

        # velocity / direction
        prev_center = t.get("center")
        velocity, direction = self._velocity_and_direction(
            prev_center, dt, center
        )
        t["velocity"]  = velocity
        t["direction"] = direction

        # odometry
        if prev_center is not None:
            seg = self._distance(prev_center, center)
            t["total_distance"] = t.get("total_distance", 0.0) + seg

        # speed stats
        speed = 0.0
        if velocity is not None:
            speed = math.hypot(velocity[0], velocity[1])
        frames = t["frames"]
        t["avg_speed"] = (
            (t.get("avg_speed", 0.0) * (frames - 1) + speed) / frames
        )
        t["max_speed"] = max(t.get("max_speed", 0.0), speed)

        # idle time
        if speed < self.idle_speed_threshold and dt > 0:
            t["idle_time"] = t.get("idle_time", 0.0) + dt

        # confidence smoothing
        confs = t.get("confidences", [])
        confs.append(conf)
        confs = confs[-self.history_size:]
        t["confidences"]    = confs
        t["avg_confidence"] = float(np.mean(confs))
        t["max_confidence"] = float(max(confs))

        # geometry
        t["center"] = center
        t["bbox"]   = bbox
        t["label"]  = label or t.get("label", "")
        t["is_equipment"] = t["label"] in EQUIPMENT_LABELS

        return t

    # -----------------------------------------------------------------------
    # Detection annotation
    # -----------------------------------------------------------------------

    @staticmethod
    def _annotate_detection(det: dict, track: dict):
        det["track_id"]       = track["id"]
        det["track_state"]    = track["state"]
        det["trajectory"]     = track["trajectory"]
        det["frames_tracked"] = track["frames"]
        det["age"]            = track["age"]
        det["velocity"]       = track["velocity"]
        det["direction"]      = track["direction"]
        det["total_distance"] = track["total_distance"]
        det["avg_speed"]      = track["avg_speed"]
        det["max_speed"]      = track["max_speed"]
        det["idle_time"]      = track["idle_time"]
        det["avg_confidence"] = track["avg_confidence"]
        det["max_confidence"] = track["max_confidence"]
        det["last_seen"]      = track["last_seen"]
        det["zone"]           = track.get("zone")
        det["is_equipment"]   = track.get("is_equipment", False)

    # -----------------------------------------------------------------------
    # Geometry helpers
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

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih

        if inter == 0:
            return 0.0

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union  = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    @staticmethod
    def _velocity_and_direction(
        prev_center,
        dt:    float,
        center: tuple,
    ):
       
        if prev_center is None or dt <= 0:
            return None, None

        vx = (center[0] - prev_center[0]) / dt
        vy = (center[1] - prev_center[1]) / dt
        direction = math.degrees(math.atan2(vy, vx))
        return (vx, vy), direction

    @staticmethod
    def _new_id() -> str:
        return str(uuid.uuid4())[:8]