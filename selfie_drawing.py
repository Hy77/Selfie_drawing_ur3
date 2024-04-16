from img_processing import ImgProcessor
from toolpathing import PathPlanner
from paper_detector import PaperDetector
from ur3_control_actionlib import UR3Control
import rospy
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber


class SelfieDrawer:
    def __init__(self, image, method, predictor_path):
        self.img_processor = ImgProcessor()
        self.path_planner = PathPlanner
        self.paper_detector = PaperDetector()
        # self.ur3_controller = UR3Control()
        self.paper_local_info = None
        self.paper_global_info = None
        self.image = image
        self.method = method
        self.predictor_path = predictor_path
        self.final_contours = []
        self.final_image = None

    def handle_img_processing(self):
        self.img_processor.img_processing(self.image, self.method, self.predictor_path)

    def final_contour_updator(self):
        self.final_contours = self.img_processor.final_contours
        self.final_image = self.img_processor.final_image

    def tsp_algo(self):
        self.final_contour_updator()  # update final_contours & image
        self.path_planner = PathPlanner(self.final_contours, self.final_image)
        self.path_planner.visualization()

    def start_drawing(self):
        print("---------------------- Moving to Init Pose -----------------------")
        # self.ur3_controller.run([0.3, -0.1, 0.47])  # init pose -> facing down to the paper
        print("Done\n--------------------- Start Paper Detecting ----------------------")
        # self.paper_local_info = self.paper_detector.paper_info  # update paper local info
        # self.paper_global_info = self.ur3_controller.define_paper_global_coord(self.paper_local_info)  # convert to glb
        print("Done\n---------------------- Start Paper Locating ----------------------")
        # self.ur3_controller.run(self.paper_global_info)  # let ur3 to reach 4 corners of the paper
        print("Done\n--------------------- Start Image Processing ---------------------")
        self.handle_img_processing()  # run img processing & get contours
        print("Done\n------------------------Start TSP Planning -----------------------")
        # self.tsp_algo()  # run tsp algo get effective path
        # TODO: control ur3 to move to these coordinates

    def callback(self, msg_color, msg_depth):
        try:
            self.paper_detector.callback(msg_color, msg_depth)
            for i in range(3, 0, -1):  # 3s to make sure its stable
                self.paper_detector.paper_detection()
            if self.paper_detector.paper_found:
                rospy.signal_shutdown('Paper found.')
                self.start_drawing()
        except Exception as e:
            print(f"Error in callback: {e}")

    def run(self):
        self.start_drawing()
        ''' # Run RGB-D camera to get paper info then start img processing & drawing etc
        rospy.init_node('selfie_drawer_node', anonymous=True)
        sub_color = Subscriber("/camera/color/image_raw", Image)
        sub_depth = Subscriber("/camera/depth/image_rect_raw", Image)
        ats = ApproximateTimeSynchronizer([sub_color, sub_depth], queue_size=5, slop=0.1)
        ats.registerCallback(self.callback)
        rospy.spin()
        '''
