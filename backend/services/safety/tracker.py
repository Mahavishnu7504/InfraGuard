import math
import uuid
import time


class EnterpriseTracker:

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

        return math.sqrt(

            (c1[0] - c2[0]) ** 2 +
            (c1[1] - c2[1]) ** 2
        )

    # =====================================
    # UPDATE
    # =====================================

    def update(
        self,
        detections
    ):

        now = time.time()

        updated_tracks = {}

        assigned = set()

        for det in detections:

            bbox = det.get(
                "bbox",
                []
            )

            if len(bbox) != 4:
                continue

            center = self._center(
                bbox
            )

            matched_id = None

            best_distance = (
                self.max_distance
            )

            for track_id, track in self.tracks.items():

                if track_id in assigned:
                    continue

                dist = self._distance(

                    center,

                    track["center"]
                )

                if dist < best_distance:

                    matched_id = track_id

                    best_distance = dist

            # =================================
            # EXISTING TRACK
            # =================================

            if matched_id:

                track = self.tracks[
                    matched_id
                ]

                trajectory = track.get(
                    "trajectory",
                    []
                )

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

                assigned.add(
                    matched_id
                )

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

            if track_id in updated_tracks:
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