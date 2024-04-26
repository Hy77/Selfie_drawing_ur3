import cv2
import numpy as np
import threading
from cv_bridge import CvBridge

class PaperDetector:
    def __init__(self):
        self.fx = 615.657  # Focal length in pixels
        self.fy = 614.627  # Focal length in pixels
        self.cx = 321.782  # Camera optical center in pixels
        self.cy = 240.117  # Camera optical center in pixels
        self.cv_bridge = CvBridge()
        self.latest_img_color = None
        self.latest_img_depth = None
        self.not_detected = True
        self.image_lock = threading.Lock()
        self.display_thread = threading.Thread(target=self.display_loop)
        self.display_thread.daemon = True
        self.display_thread.start()

    def display_loop(self):
        while self.not_detected:
            with self.image_lock:
                if self.latest_img_color is not None:
                    self.latest_img_color = self.draw_camera_axes(self.latest_img_color)
                    cv2.imshow('RGB_Img', self.latest_img_color)
                    cv2.waitKey(1)
            key = cv2.waitKey(1)
            if key == 27:  # Exit on ESC key
                self.stop_display()  # Stop display and clean up
            elif key == 13:  # Exit on Enter key
                self.stop_display()  # Stop display and clean up

    def stop_display(self):
        self.not_detected = False
        cv2.destroyAllWindows()  # Close all OpenCV windows

    @staticmethod
    def draw_camera_axes(image):
        # 确定坐标轴的起点（图像中的位置）
        origin = (50, 50)  # 假设坐标系原点在图像的(50, 50)处

        # 定义坐标轴长度
        axis_length = 40

        # X轴：红色
        x_axis_end = (origin[0] + axis_length, origin[1])
        cv2.arrowedLine(image, origin, x_axis_end, (0, 0, 255), 2)

        # Y轴：绿色
        y_axis_end = (origin[0], origin[1] + axis_length)
        cv2.arrowedLine(image, origin, y_axis_end, (0, 255, 0), 2)

        # 在轴线旁边添加标签
        cv2.putText(image, 'X', (x_axis_end[0] + 10, x_axis_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(image, 'Y', (y_axis_end[0] - 10, y_axis_end[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image

    @staticmethod
    def order_points(pts):
        # 根据x+y的值找到右下角的点
        s = [pt[0] + pt[1] for pt in pts]
        bottom_right_index = np.argmax(s)
        ordered = pts[bottom_right_index:] + pts[:bottom_right_index]

        # 确保是逆时针方向
        # 计算向量叉乘，判断顺序（根据前三个点）
        if len(ordered) >= 3:
            vec1 = np.array(ordered[1]) - np.array(ordered[0])
            vec2 = np.array(ordered[2]) - np.array(ordered[1])
            cross_product = np.cross(vec1, vec2)
            if cross_product > 0:
                # 如果是顺时针，反转顺序（除了第一个点）
                ordered[1:] = ordered[1:][::-1]
        return ordered

    @staticmethod
    def swap_y_values(global_corners):
        # 提取x, y, z值
        x_values, y_values, z_values = zip(*global_corners)

        # 交换y值
        y_values = list(y_values)
        y_values[0], y_values[1] = y_values[1], y_values[0]
        y_values[2], y_values[3] = y_values[3], y_values[2]

        # 重建全局角点列表
        swapped_corners = [(x, y, z) for x, y, z in zip(x_values, y_values, z_values)]
        return swapped_corners
    '''
    0.02089069917909729 0.16317639228163522
    0.02089069917909729 -0.14971685319771938
    -0.20226560341936545 -0.14971685319771938
    -0.20226560341936545 0.16317639228163522
    glb [(-0.138885716968896, 0.08911164722355418, 0.005394941181328383), 
    (-0.4517789624482506, 0.08911164722355418, 0.005394941181328383), 
    (-0.4517789624482506, 0.3122679498220169, 0.005394941181328383), 
    (-0.138885716968896, 0.3122679498220169, 0.005394941181328383)]
    Final detected paper info: {'corners': [(-0.138885716968896, 0.08911164722355418, 0.005394941181328383), (-0.4517789624482506, 0.08911164722355418, 0.005394941181328383), (-0.4517789624482506, 0.3122679498220169, 0.005394941181328383), (-0.138885716968896, 0.3122679498220169, 0.005394941181328383)]}
    '''
    def calculate_local_coordinates(self, camera_pose, corners, mean_depth):
        fx = 912.508056640625
        fy = 912.2136840820312
        cx = 651.252197265625
        cy = 348.5895080566406

        global_corners = []
        camera_x, camera_y, camera_z = camera_pose
        # (-0.30205597795739236, 0.11000992991402664, 0.43140717990593264)
        # print(camera_pose)
        for (x_pixel, y_pixel) in corners:
            # 将像素坐标转换为以摄像头为原点的空间坐标
            x_local = (x_pixel - cx) * mean_depth / fx
            y_local = (y_pixel - cy) * mean_depth / fy
            # print(x_local, y_local)
            # UR3坐标系调整: 交换x和y
            x_ur3 = camera_x + y_local
            y_ur3 = camera_y + x_local - 0.02806  # 0.02806 -> we defined the camera_x in ur3_control_actionlib.py
            z_ur3 = camera_z - mean_depth  # camera facing down -> negative mean depth

            # 将坐标添加到列表
            global_corners.append((x_ur3, y_ur3, z_ur3))
        # print(global_corners)
        # return self.swap_y_values(global_corners)
        return global_corners
    def paper_detection(self, cam_ee_pose):
        with self.image_lock:
            if self.latest_img_color is not None and self.latest_img_depth is not None:
                hsv_image = cv2.cvtColor(self.latest_img_color, cv2.COLOR_BGR2HSV)
                sen = 75
                lower_white = np.array([25, 0, 150])
                upper_white = np.array([200, 80, 255])
                mask = cv2.inRange(hsv_image, lower_white, upper_white)
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_contour) > 200000:
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        cv2.rectangle(self.latest_img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        mask = np.zeros_like(self.latest_img_depth, dtype=np.uint8)
                        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
                        masked_depth = cv2.bitwise_and(self.latest_img_depth, self.latest_img_depth, mask=mask)
                        valid_depths = masked_depth[masked_depth > 0]
                        mean_depth = np.mean(valid_depths) if valid_depths.size > 0 else 0
                        mean_depth_meters = (mean_depth - 10) / 1000.0  # Convert mm to meters

                        # Define corners based on bounding box
                        corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                        ordered_corners = self.order_points(corners)
                        for i, corner in enumerate(ordered_corners):
                            cv2.circle(self.latest_img_color, corner, 5, (0, 0, 255), -1)
                            cv2.putText(self.latest_img_color, str(i), corner, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0),
                                        2)
                        # print(cam_ee_pose)
                        glb_points = self.calculate_local_coordinates(cam_ee_pose, ordered_corners, mean_depth_meters)
                        # print('glb', glb_points)

                        # local_corners = [((corner[0]) * mean_depth_meters,
                        #                   (corner[1]) * mean_depth_meters,
                        #                   mean_depth_meters) for corner in corners]

                        self.paper_info = {
                            'corners': glb_points,
                        }

    def callback(self, msg_color, msg_depth):
        with self.image_lock:
            self.latest_img_color = self.cv_bridge.imgmsg_to_cv2(msg_color, "bgr8")
            self.latest_img_depth = self.cv_bridge.imgmsg_to_cv2(msg_depth, "32FC1")
