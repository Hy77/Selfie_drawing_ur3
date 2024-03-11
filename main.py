#!/usr/bin/env python3
# main.py
import cv2

from selfie_drawing import SelfieDrawer


def main():
    selfie_drawer = SelfieDrawer()
    try:
        method = input(
            "Choose a vectorization method ('n' for nearest neighbor, 'b' for bilinear, or 'c' for bicubic): ").lower()
        size = input(
            "Choose a img size (enter 'a3' or press 'enter' to use origin img size): ").lower()
        image = cv2.imread('photo_1.jpg')
        selfie_drawer.run(image=image, method=method, size=size)
    except KeyboardInterrupt:
        print("Shutting down")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
