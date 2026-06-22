import cv2
from ai_engine.pipelines.pipeline import InfraGuardPipeline


class VideoMonitor:

    def __init__(self, video_source):

        self.video_source = video_source
        self.pipeline = InfraGuardPipeline()

    def run(self):

        cap = cv2.VideoCapture(self.video_source)

        if not cap.isOpened():
            raise RuntimeError("Unable to open video source")

        frame_id = 0

        print("InfraGuard monitoring started... Press Q to exit")

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            result = self.pipeline.process_frame(frame, frame_id)

            processed = result["output_image"]

            cv2.imshow("InfraGuard Monitor", processed)

            frame_id += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    monitor = VideoMonitor("data/test_video/ppe.mp4")
    monitor.run()