from img_processing import ImgProcessor
from toolpathing import PathPlanner
from paper_detector import PaperDetector
import rospy
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber

class SelfieDrawer:
    def __init__(self, image, method, size, predictor_path):
        self.img_processor = ImgProcessor()
        self.path_planner = PathPlanner
        self.paper_detector = PaperDetector()
        self.image = image
        self.method = method
        self.size = size
        self.predictor_path = predictor_path
        self.final_contours = []
        self.final_image = None

    def handle_img_processing(self):
        self.img_processor.img_processing(self.image, self.method, self.size, self.predictor_path)

    def final_contour_updator(self):
        self.final_contours = self.img_processor.final_contours
        self.final_image = self.img_processor.final_image

    def tsp_algo(self):
        self.final_contour_updator()  # update final_contours & image
        self.path_planner = PathPlanner(self.final_contours, self.final_image)
        self.path_planner.visualization()

    def callback(self, msg_color, msg_depth):
        try:
            self.paper_detector.callback(msg_color, msg_depth)
            self.paper_detector.paper_detection()
            if self.paper_detector.paper_found:
                rospy.signal_shutdown('Paper found.')
                self.handle_img_processing()
                self.final_contour_updator()
                self.tsp_algo()
        except Exception as e:
            print(f"Error in callback: {e}")

    def run(self):
        rospy.init_node('selfie_drawer_node', anonymous=True)
        sub_color = Subscriber("/camera/color/image_raw", Image)
        sub_depth = Subscriber("/camera/depth/image_rect_raw", Image)
        ats = ApproximateTimeSynchronizer([sub_color, sub_depth], queue_size=5, slop=0.1)
        ats.registerCallback(self.callback)
        rospy.spin()
