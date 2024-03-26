import cv2
import numpy as np
import threading
from cv_bridge import CvBridge


class PaperDetector:
    def __init__(self):
        self.fx = 615.657
        self.fy = 614.627
        self.cx = 321.782
        self.cy = 240.117
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
                # Convert to HSV
                hsv_image = cv2.cvtColor(self.latest_img_color, cv2.COLOR_BGR2HSV)

                # Define sensitivity and white color range
                sen = 90
                lower_white = np.array([0, 0, 255 - sen])
                upper_white = np.array([255, sen, 255])

                # Create mask and filter noise
                mask = cv2.inRange(hsv_image, lower_white, upper_white)
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                # Find contours and filter by area
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                min_area = 500  # Set threshold
                large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

                if large_contours:
                    # Find largest contour and draw it
                    largest_contour = max(large_contours, key=cv2.contourArea)
                    cv2.drawContours(self.latest_img_color, [largest_contour], -1, (0, 255, 0), 2)

                    # Find corners
                    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                    corners = cv2.approxPolyDP(largest_contour, epsilon, True)

                    # Label corners
                    for corner in corners:
                        cv2.circle(self.latest_img_color, tuple(corner[0]), 5, (0, 0, 255), -1)

                    # Calculate the real size of the A4 paper (in mm)
                    real_width = 210  # A4 paper width in mm
                    real_height = 297  # A4 paper height in mm

                    # Calculate the detected size in pixels
                    x, y, w, h = cv2.boundingRect(largest_contour)

                    # Calculate the distance based on the detected width and the real width of the A4 paper
                    distance_width = (self.fx * real_width) / w

                    # Calculate the distance based on the detected height and the real height of the A4 paper
                    distance_height = (self.fy * real_height) / h

                    # Take the average of the two distances as the final estimate
                    distance = (distance_width + distance_height) / 2  # in mm

                    # Calculate the real dimensions of the detected paper
                    real_detected_width = distance * w / self.fx
                    real_detected_height = distance * h / self.fy

                    # Display the real dimensions on the edges of the detected paper
                    cv2.putText(self.latest_img_color, f"Real Width: {real_detected_width:.2f}mm", (x + w // 2, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(self.latest_img_color, f"Real Height: {real_detected_height:.2f}mm",
                                (x - 30, y + h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    # Display the distance at the center of the detected paper
                    center_x, center_y = x + w // 2, y + h // 2
                    cv2.putText(self.latest_img_color, f"Distance: {distance:.2f}mm", (center_x - 50, center_y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    # Convert corners to local coordinates
                    local_corners = []
                    for corner in corners:
                        u, v = corner[0]
                        x_coord = (u - self.cx) * distance / self.fx
                        y_coord = (v - self.cy) * distance / self.fy
                        local_corners.append((x_coord, y_coord, distance))

                    # Store paper information
                    self.paper_info = {
                        'corners': local_corners,
                        'distance': distance
                    }

    def callback(self, msg_color, msg_depth):
        with self.image_lock:
            self.latest_img_color = self.cv_bridge.imgmsg_to_cv2(msg_color, "bgr8")
            self.latest_img_depth = self.cv_bridge.imgmsg_to_cv2(msg_depth, "32FC1")

