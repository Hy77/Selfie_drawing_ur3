import cv2
import numpy as np
import threading
from cv_bridge import CvBridge


class PaperDetector:
    def __init__(self):
        self.cv_bridge = CvBridge()
        self.latest_img_color = None
        self.latest_img_depth = None
        self.image_lock = threading.Lock()
        self.paper_found = False
        self.paper_info = None
        self.display_thread = threading.Thread(target=self.display_loop)
        self.display_thread.daemon = True
        self.display_thread.start()

    def display_loop(self):
        while True:
            with self.image_lock:
                if self.latest_img_color is not None:
                    cv2.imshow('RGB_Img', self.latest_img_color)
                    cv2.waitKey(1)

    def paper_detection(self):
        with self.image_lock:
            if self.latest_img_color is not None:
                # convert to hsv
                hsv_image = cv2.cvtColor(self.latest_img_color, cv2.COLOR_BGR2HSV)

                # define WHITE
                lower_white = np.array([0, 0, 200])
                upper_white = np.array([180, 55, 255])

                # create mask
                mask = cv2.inRange(hsv_image, lower_white, upper_white)

                # filter nosie
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                # find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # area filter
                min_area = 500  # set threshold
                large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

                if large_contours:
                    # find largest contour
                    largest_contour = max(large_contours, key=cv2.contourArea)
                    cv2.drawContours(self.latest_img_color, [largest_contour], -1, (0, 255, 0), 2)

                    # find corners
                    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                    corners = cv2.approxPolyDP(largest_contour, epsilon, True)

                    # label corners
                    for corner in corners:
                        cv2.circle(self.latest_img_color, tuple(corner[0]), 5, (0, 0, 255), -1)

                    # store paper's info
                    self.paper_info = {
                        'corners': corners,
                        'depth': None
                    }

                    # get xywh
                    x, y, w, h = cv2.boundingRect(largest_contour)

                    # if got 4 points then cal depth
                    if len(corners) == 4:
                        depth_values = []
                        for corner in corners:
                            x, y = corner[0]
                            depth = self.latest_img_depth[y, x]
                            depth_values.append(depth)
                        average_depth = np.mean(depth_values)
                        self.paper_info['depth'] = average_depth

                        # show depth info on img
                        cv2.putText(self.latest_img_color, f"Depth: {average_depth:.2f} mm",
                                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # self.paper_found = True

            # print("Displayed RGB image with paper contour and depth")

    def callback(self, msg_color, msg_depth):
        with self.image_lock:
            self.latest_img_color = self.cv_bridge.imgmsg_to_cv2(msg_color, "bgr8")
            self.latest_img_depth = self.cv_bridge.imgmsg_to_cv2(msg_depth, "32FC1")

