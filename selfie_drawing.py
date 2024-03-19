#!/usr/bin/env python3
# selfie_drawing.py

from img_processing import ImgProcessor


class SelfieDrawer():
    def __init__(self):
        self.img_processor = ImgProcessor()
        self.vector = None
        self.tsp_vector = None
        self.final_contours = []
        self.final_image = None

    def handle_img_processing(self, image, method, size, predictor_path):
        self.img_processor.img_processing(image, method, size, predictor_path)

    def final_contour_updator(self):
        self.final_contours = self.img_processor.final_contours
        self.final_image = self.img_processor.final_image

    def tsp_algo(self):
        return self.tsp_vector

    def ur3_controller(self):
        return

    def run(self, image, method, size, predictor_path):
        self.handle_img_processing(image, method, size, predictor_path)
