import cv2
import time
import logging
import threading
import asyncio
from cvzone.FaceMeshModule import FaceMeshDetector

import time

logging.basicConfig(level=logging.INFO)

class ScreenToFaceDistance:
    def __init__(self, interval: float, focal_length: float, real_width: float, display: bool = False):
        self.interval = interval
        self.focal_length = focal_length
        self.real_width = real_width
        self.delta = 0
        self.display = display
        self.total_time = 0
        self.camera = cv2.VideoCapture(0)
        self.face_detector = FaceMeshDetector(maxFaces=1)

    def calculate_point_distance(self, point1, point2) -> float:
        return self.face_detector.findDistance(point1, point2)[0]

    def capture_frame(self):
        success, frame = self.camera.read()
        if not success:
            logging.error("Failed to capture frame")
            return None
        return cv2.flip(frame, 1)

    def detect_face_landmarks(self, frame):
        frame, faces = self.face_detector.findFaceMesh(frame, draw=False)
        if not faces:
            print("No face detected")
            return None, None
        face = faces[0]
        left_eye = face[23]
        right_eye = face[374]
        return left_eye, right_eye

    def compute_face_distance(self, left_eye, right_eye) -> float:
        eye_distance = self.calculate_point_distance(left_eye, right_eye)
        return (self.real_width * self.focal_length) / eye_distance

    def analyze_frame(self):
        frame = self.capture_frame()
        if frame is None:
            return

        left_eye, right_eye = self.detect_face_landmarks(frame)
        if left_eye and right_eye:
            self.total_time += self.interval
            cv2.circle(frame, left_eye, 5, (0, 0, 0), cv2.FILLED)
            cv2.circle(frame, right_eye, 5, (0, 0, 0), cv2.FILLED)

            distance = self.compute_face_distance(left_eye, right_eye)
            logging.info(f"Distance: {distance}")
            if distance < 50:
                self.delta += self.interval

            # Display the live capture if enabled
            if self.display:
                self.show_frame(frame)
        time.sleep(self.interval)

    def show_frame(self, frame):
        cv2.imshow("Image", frame)
        cv2.waitKey(1)

    async def rule_20_20_20(self):
        while True:
            await asyncio.sleep(1200)
            print('Your eyes should take a 20 second break from the screen')
            await asyncio.sleep(20)
            print('That should be enough')

    def start(self):
        threading.Thread(target=lambda: asyncio.run(self.rule_20_20_20()), daemon=True).start()
        try:
            while True:
                self.analyze_frame()
                self.rule_20_20_20()
        except KeyboardInterrupt:
            logging.info("Stopped by user")
        finally:
            self.camera.release()
            cv2.destroyAllWindows()
            logging.info(f"Out of a total of {round(self.total_time, 1)} seconds, {round(self.delta, 1)} seconds have been spent sitting to close to the screen")