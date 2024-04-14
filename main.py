#!/usr/bin/env python3
# main.py
import cv2
import time
from selfie_drawing import SelfieDrawer

import rospkg

# find pkg path
rospack = rospkg.RosPack()
package_path = rospack.get_path('selfie_drawer_pkg')

# get img path based on the pkg path
predictor_path = package_path + '/Selfie_drawing_ur3/shape_predictor_68_face_landmarks.dat'


def take_photo():
    cap = cv2.VideoCapture(0)  # Open the default camera
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Camera", 850, 600)
    print("Press Enter to start the countdown...")  # Print the message once

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('\r'):  # Check if Enter is pressed
                for i in range(3, 0, -1):  # Countdown from 3 to 1
                    print(i)
                    start_time = time.time()
                    # Show the live camera feed for 1 second
                    while time.time() - start_time < 1:
                        ret, frame = cap.read()
                        if ret:
                            cv2.imshow("Camera", frame)
                            cv2.waitKey(1)
                cv2.imwrite('captured_photo.jpg', frame)  # Save the captured image
                break
        else:
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all OpenCV windows
    if ret:
        return frame
    else:
        raise Exception("Failed to capture photo")


def main():
    method = input(
        "Choose a vectorization method ('n' for nearest neighbor, 'b' for bilinear, or 'c' for bicubic): ").lower()
    # image = take_photo()  # Capture a photo from the camera
    image = cv2.imread(package_path + '/Selfie_drawing_ur3/photo_1.jpg')    # For debugging
    if image is None:
        raise FileNotFoundError(f"Image file captured_photo.jpg not found.")
    selfie_drawer = SelfieDrawer(image, method, predictor_path)
    selfie_drawer.run()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
