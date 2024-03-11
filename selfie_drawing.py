#!/usr/bin/env python3
# selfie_drawing.py

from img_processing import ImgProcessor


class SelfieDrawer():
    def __init__(self):
        self.img_processor = ImgProcessor()
        self.vector = None
        self.tsp_vector = None

    def handle_img_processing(self, image, method, size):
        self.img_processor.img_processing(image, method, size)

    def tsp_algo(self):
        return self.tsp_vector

    def ur3_controller(self):
        return

    def run(self, image, method, size):
        self.handle_img_processing(image, method, size)
