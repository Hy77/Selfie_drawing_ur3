import cv2
import dlib
import numpy as np


class SenseDetector:
    def __init__(self):
        # Initialize Dlib's face detector and landmark predictor
        self.predictor = None
        self.detector = dlib.get_frontal_face_detector()

    def detect_senses(self, predictor_path, gray_img):
        # Define predictor
        self.predictor = dlib.shape_predictor(predictor_path)

        # Detect faces in the image
        faces = self.detector(gray_img)

        # Create a mask for the facial features
        facial_features_mask = np.zeros_like(gray_img)

        # Initialize a list to store facial feature contours
        facial_contours = []

        # Iterate over faces
        for face in faces:
            landmarks = self.predictor(gray_img, face)

            # Draw the facial feature points on the mask
            for n in range(68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(facial_features_mask, (x, y), 2, (255, 255, 255), -1)

            # Extract facial feature contours
            facial_contours.append(self.extract_contours(landmarks))

        return facial_contours

    @staticmethod
    def extract_contours(landmarks):
        contours = {'jaw': [(landmarks.part(n).x, landmarks.part(n).y) for n in range(0, 17)],
                    'left_eyebrow': [(landmarks.part(n).x, landmarks.part(n).y) for n in range(17, 22)],
                    'right_eyebrow': [(landmarks.part(n).x, landmarks.part(n).y) for n in range(22, 27)],
                    'nose': [(landmarks.part(n).x, landmarks.part(n).y) for n in range(27, 36)],
                    'left_eye': [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)],
                    'right_eye': [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)],
                    'outer_lip': [(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 60)],
                    'inner_lip': [(landmarks.part(n).x, landmarks.part(n).y) for n in range(60, 68)]}
        return contours
