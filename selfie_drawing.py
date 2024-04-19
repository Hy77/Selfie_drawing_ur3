from img_processing import ImgProcessor
from toolpathing import PathPlanner
from paper_detector import PaperDetector
from ur3_control_actionlib import UR3Control
import rospy
import time as timer
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber


class SelfieDrawer:
    def __init__(self, image, method, predictor_path):
        self.ur3_controller = UR3Control()
        self.img_processor = ImgProcessor()
        self.path_planner = PathPlanner
        self.paper_detector = PaperDetector()
        self.paper_local_info = None
        self.paper_global_info = None
        self.image = image
        self.method = method
        self.predictor_path = predictor_path
        self.final_contours = []
        self.final_image = None
        self.paper_detected = False
        self.detection_attempts = 0

    def handle_img_processing(self):
        self.img_processor.img_processing(self.image, self.method, self.predictor_path)

    def final_contour_updator(self):
        self.final_contours = self.img_processor.final_contours
        self.final_image = self.img_processor.final_image

    def tsp_algo(self):
        self.final_contour_updator()  # update final_contours & image
        self.path_planner = PathPlanner(self.final_contours, self.final_image)
        self.path_planner.visualization(600, 850)

    def start_drawing(self):
        print("Done\n---------------------- Start Paper Location ----------------------")
        self.ur3_controller.run(self.paper_local_info['corners'][0:], 0.23)  # let ur3 to reach 4 corners of the paper
        print("Done\n--------------------- Start Image Processing ---------------------")
        self.handle_img_processing()  # run img processing & get contours
        print("Done\n------------------------Start TSP Planning -----------------------")
        # self.tsp_algo()  # run tsp algo get effective path
        # TODO: control ur3 to move to these coordinates

    def callback(self, msg_color, msg_depth):
        try:
            if not self.paper_detected:
                self.paper_detector.callback(msg_color, msg_depth)
                # 开始3秒定时检测
                self.timer = rospy.Timer(rospy.Duration(1), self.timer_callback, oneshot=False)
            self.start_drawing()
        except Exception as e:
            print(f"Error in callback: {e}")

    def timer_callback(self, event):
        if self.detection_attempts < 5:
            self.paper_detector.paper_detection(self.ur3_controller.get_current_camera_ee_xyz())
            self.paper_local_info = self.paper_detector.paper_info  # update paper local info
            self.detection_attempts += 1
            # print("Detection attempt", self.detection_attempts)
        else:
            self.timer.shutdown()  # 停止定时器
            self.paper_detector.display_loop(False)
            # print("Final detected paper info:", self.paper_local_info)
            if len(self.paper_local_info['corners']) != 0:
                self.paper_detected = True

    def run(self):
        print("---------------------- Moving to Init Pose -----------------------")
        self.ur3_controller.run([(-0.274, 0.11, 0.485)], 0)  # init pose -> facing down to the paper
        print("Done\n--------------------- Start Paper Detection ----------------------")
        # Run RGB-D camera to get paper info then start img processing & drawing etc
        sub_color = Subscriber("/camera/color/image_raw", Image)
        sub_depth = Subscriber("/camera/depth/image_rect_raw", Image)
        ats = ApproximateTimeSynchronizer([sub_color, sub_depth], queue_size=5, slop=0.1)
        ats.registerCallback(self.callback)
        rospy.spin()

