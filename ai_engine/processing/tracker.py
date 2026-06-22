import math


class  EnterpriseTracker:
    """
    Lightweight object tracker that assigns IDs to detections
    based on spatial proximity.
    """

    def __init__(self, max_distance=50):

        self.next_id = 1
        self.tracks = {}
        self.max_distance = max_distance

    def _center(self, bbox):

        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _distance(self, c1, c2):

        return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

    def update(self, detections):

        updated_tracks = {}
        results = []

        for det in detections:

            center = self._center(det["bbox"])

            matched_id = None

            for track_id, track_center in self.tracks.items():

                if self._distance(center, track_center) < self.max_distance:
                    matched_id = track_id
                    break

            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1

            updated_tracks[matched_id] = center

            det["track_id"] = matched_id

            results.append(det)

        self.tracks = updated_tracks

        return results