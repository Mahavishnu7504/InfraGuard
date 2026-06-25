import math
import uuid
import time

import numpy as np
from scipy.optimize import linear_sum_assignment


class EnterpriseTracker:
    """
    Simple multi-object tracker using global nearest-neighbor assignment
    (Hungarian algorithm) instead of greedy per-detection matching.

    Why global assignment instead of greedy matching:
    Greedy "for each detection, pick the closest free track" assignment
    is order-dependent. When two tracked objects cross paths or come
    close together, a detection processed earlier can steal the track
    that "belongs" to a detection processed later, causing an ID swap.
    Solving the full assignment problem (minimizing total distance
    across all detection-track pairs simultaneously) avoids this.

    Note on max_missing:
    This is compared against `time.time()` deltas, so it is a number
    of SECONDS a track may go undetected before being dropped, not a
    number of frames. If you want frame-count semantics instead, pass
    in a frame counter rather than wall-clock time.
    """

    def __init__(
        self,
        max_distance=90,
        max_missing=20,
        history_size=35
    ):

        self.tracks = {}

        self.max_distance = max_distance
        self.max_missing = max_missing
        self.history_size = history_size

    # =====================================
    # CENTER
    # =====================================

    def _center(self, bbox):

        x1, y1, x2, y2 = bbox

        return (
            int((x1 + x2) / 2),
            int((y1 + y2) / 2)
        )

    # =====================================
    # DISTANCE
    # =====================================

    def _distance(self, c1, c2):

        return math.hypot(
            c1[0] - c2[0],
            c1[1] - c2[1]
        )

    # =====================================
    # ASSIGNMENT (Hungarian / global nearest-neighbor)
    # =====================================

    def _assign(self, det_centers, track_ids):
        """
        Solve the assignment problem between detections and existing
        tracks, minimizing total distance. Returns a dict mapping
        detection index -> track_id for every match that is within
        max_distance. Detections/tracks with no acceptable match are
        simply absent from the returned dict.
        """

        if not det_centers or not track_ids:
            return {}

        cost = np.zeros(
            (len(det_centers), len(track_ids))
        )

        for i, center in enumerate(det_centers):
            for j, track_id in enumerate(track_ids):
                cost[i, j] = self._distance(
                    center,
                    self.tracks[track_id]["center"]
                )

        det_idx, track_idx = linear_sum_assignment(cost)

        matches = {}

        for d_i, t_i in zip(det_idx, track_idx):

            if cost[d_i, t_i] < self.max_distance:
                matches[d_i] = track_ids[t_i]

        return matches

    # =====================================
    # UPDATE
    # =====================================

    def update(
        self,
        detections
    ):

        now = time.time()

        updated_tracks = {}

        # Pre-filter detections with valid bboxes, keep original
        # detections list intact (including invalid ones) since
        # we still return the full `detections` list at the end.
        valid_indices = []
        det_centers = []

        for i, det in enumerate(detections):

            bbox = det.get(
                "bbox",
                []
            )

            if len(bbox) != 4:
                continue

            valid_indices.append(i)
            det_centers.append(
                self._center(bbox)
            )

        track_ids = list(self.tracks.keys())

        matches = self._assign(
            det_centers,
            track_ids
        )

        assigned_track_ids = set(
            matches.values()
        )

        for local_i, original_i in enumerate(valid_indices):

            det = detections[original_i]
            bbox = det.get("bbox")
            center = det_centers[local_i]

            matched_id = matches.get(local_i)

            # =================================
            # EXISTING TRACK
            # =================================

            if matched_id is not None:

                track = self.tracks[
                    matched_id
                ]

                # Copy before mutating so the old track dict (if
                # referenced elsewhere) isn't mutated in place.
                trajectory = track.get(
                    "trajectory",
                    []
                )[:]

                trajectory.append(
                    center
                )

                trajectory = trajectory[
                    -self.history_size:
                ]

                frames = (
                    track.get(
                        "frames",
                        0
                    ) + 1
                )

                updated_tracks[
                    matched_id
                ] = {

                    "id":
                        matched_id,

                    "center":
                        center,

                    "bbox":
                        bbox,

                    "trajectory":
                        trajectory,

                    "first_seen":
                        track.get(
                            "first_seen",
                            now
                        ),

                    "last_seen":
                        now,

                    "frames":
                        frames
                }

                det["track_id"] = (
                    matched_id
                )

                det["trajectory"] = (
                    trajectory
                )

                det["frames_tracked"] = (
                    frames
                )

            # =================================
            # NEW TRACK
            # =================================

            else:

                new_id = str(
                    uuid.uuid4()
                )[:8]

                updated_tracks[
                    new_id
                ] = {

                    "id":
                        new_id,

                    "center":
                        center,

                    "bbox":
                        bbox,

                    "trajectory":
                        [center],

                    "first_seen":
                        now,

                    "last_seen":
                        now,

                    "frames":
                        1
                }

                det["track_id"] = (
                    new_id
                )

                det["trajectory"] = [
                    center
                ]

                det["frames_tracked"] = 1

        # =================================
        # KEEP MISSING TRACKS
        # =================================

        for track_id, track in self.tracks.items():

            if track_id in assigned_track_ids:
                continue

            last_seen = track.get(
                "last_seen",
                now
            )

            if (
                now - last_seen
            ) < self.max_missing:

                updated_tracks[
                    track_id
                ] = track

        self.tracks = updated_tracks

        return detections