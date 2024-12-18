from function import ScreenToFaceDistance
from constants import INTERVAL, FOCAL_LENGTH, REAL_WIDTH

if __name__ == "__main__":
    screen_to_face_distance = ScreenToFaceDistance(INTERVAL, FOCAL_LENGTH, REAL_WIDTH, display=True)
    screen_to_face_distance.start()
