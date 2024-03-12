#!/usr/bin/env python3
# main.py
import cv2

from selfie_drawing import SelfieDrawer

import rospkg

# find pkg path
rospack = rospkg.RosPack()
package_path = rospack.get_path('selfie_drawer_pkg')

# get img path based on the pkg path
image_path = package_path + '/Selfie_drawing_ur3/photo_1.jpg'
predictor_path = package_path + '/Selfie_drawing_ur3/shape_predictor_68_face_landmarks.dat'

def main():
    selfie_drawer = SelfieDrawer()
    try:
        method = input(
            "Choose a vectorization method ('n' for nearest neighbor, 'b' for bilinear, or 'c' for bicubic): ").lower()
        size = input(
            "Choose a img size (enter 'a3' or press 'enter' to use origin img size): ").lower()
        image = cv2.imread(image_path)  # read the img
        selfie_drawer.run(image=image, method=method, size=size, predictor_path=predictor_path)
    except KeyboardInterrupt:
        print("Shutting down")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
