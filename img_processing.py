#!/usr/bin/env python3
# img_processing.py

import cv2
import numpy as np
from scipy.interpolate import interp1d, splprep, splev
from PIL import Image
from sense_detector import SenseDetector
import cv2.ximgproc as ximgproc
import rembg


class ImgProcessor():

    def __init__(self):
        self.sense_detector = SenseDetector()
        self.image = None
        self.non_facial_contours = []
        self.n_vector = None
        self.b_vector = None
        self.c_vector = None
        self.final_contours = []
        self.final_image = None

    # Use can define the img processing method
    def vectorize_contour(self, contours, method='n'):
        vectorized_contours = []
        for contour in contours:
            if method == 'n':  # nearest neighbor
                vectorized_contours.append(self.contour_to_nearest_neighbor_vector(contour))
            elif method == 'b':  # bilinear
                vectorized_contours.append(self.contour_to_bilinear_vector(contour))
            elif method == 'c':  # bicubic
                vectorized_contours.append(self.contour_to_cubic_spline(contour))
            else:
                raise ValueError(
                    "Invalid vectorization method. Choose 'n' for nearest neighbor, 'b' for bilinear, or 'c' for bicubic.")
        return vectorized_contours

    @staticmethod
    def remove_background(image):
        # Convert the input image to a numpy array
        input_array = np.array(image)
        # Apply background removal using rembg
        output_array = rembg.remove(input_array)
        # Convert RGBA output to BGR
        image_bkg_removal = cv2.cvtColor(output_array, cv2.COLOR_RGBA2BGR)

        return image_bkg_removal

    @staticmethod
    def cv2_imshow_resize_image_for_display(image_name, image):
        cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(image_name, 600, 850)
        cv2.imshow(image_name, image)

    @staticmethod
    def convert_contours_to_a5_size(original_contours, original_size):
        a5_size = [1748, 2480]
        # Calculate scale ratios in x and y dimensions
        scale_x = a5_size[0] / original_size[1]
        scale_y = a5_size[1] / original_size[0]
        scale_ratio = min(scale_x, scale_y)

        # Calculate the offsets to center the contours
        offset_x = (a5_size[0] - (original_size[1] * scale_ratio)) / 2
        offset_y = (a5_size[1] - (original_size[0] * scale_ratio)) / 2

        a5_contours = []
        for contour in original_contours:
            # Check if the contour is in the expected structure (list of lists)
            if isinstance(contour, np.ndarray) and len(contour.shape) == 3 and contour.shape[1] == 1:
                # Reshape the contour to (n, 2)
                contour = contour.reshape(-1, 2)

            scaled_contour = []
            for point in contour:
                # Apply the scaling and offset to each point
                new_point = [int(point[0] * scale_ratio + offset_x), int(point[1] * scale_ratio + offset_y)]
                scaled_contour.append(new_point)

            # Reshape the scaled contour back to (n, 1, 2) for OpenCV functions
            a5_contours.append(np.array(scaled_contour, dtype=np.int32).reshape(-1, 1, 2))

        return a5_contours

    @staticmethod
    def resize_with_aspect_ratio(image, target_width, target_height):
        # Get the dimensions of the original image
        original_height, original_width = image.shape[:2]

        # Calculate the target aspect ratio
        target_aspect_ratio = target_width / target_height

        # Calculate the aspect ratio of the original image
        original_aspect_ratio = original_width / original_height

        if original_aspect_ratio > target_aspect_ratio:
            # If the original image is wider than the target aspect ratio, scale by width
            new_width = target_width
            new_height = int(target_width / original_aspect_ratio)
        else:
            # If the original image is taller than the target aspect ratio, scale by height
            new_height = target_height
            new_width = int(target_height * original_aspect_ratio)

        # Resize the image to the new dimensions
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Create a new image with the target size and fill it with black
        if len(image.shape) == 3 and image.shape[2] == 3:
            result_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        else:
            result_image = np.zeros((target_height, target_width), dtype=np.uint8)

        # Calculate top-left corner to center the resized image
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2

        # Place the resized image on the black background
        if len(image.shape) == 3 and image.shape[2] == 3:
            result_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width, :] = resized_image
        else:
            result_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

        # Convert to color if the original image is grayscale
        if len(image.shape) == 2:
            result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)

        return result_image

    def contour_to_nearest_neighbor_vector(self, contour, merge_distance=10):
        # Convert contour coordinates to numpy array
        contour = np.array(contour, dtype=np.float32).reshape(-1, 2)

        # Merge points that are too close together
        simplified_contour = [contour[0]]
        for point in contour[1:]:
            if np.linalg.norm(simplified_contour[-1] - point) >= merge_distance:
                simplified_contour.append(point)

        final_contour = [simplified_contour[0]]
        for i in range(1, len(simplified_contour)):
            final_contour.append(simplified_contour[i])
            if i < len(simplified_contour) - 1:  # Connecting each point to the next
                final_contour.append(simplified_contour[i + 1])

        return np.array(final_contour, dtype=np.int32).reshape(-1, 1, 2)

    def contour_to_bilinear_vector(self, contour, num_points=1000, merge_distance=10):
        # Convert contour coordinates to numpy array
        contour = np.array(contour, dtype=np.float32).reshape(-1, 2)

        # Creating parameterised variables
        t = np.linspace(0, 1, len(contour[:, 0]))
        t_new = np.linspace(0, 1, num_points)

        # quadratic interpolation function
        f_x = interp1d(t, contour[:, 0], kind='quadratic')
        f_y = interp1d(t, contour[:, 1], kind='quadratic')

        # Combine new x and y coordinate points
        x_new = f_x(t_new)
        y_new = f_y(t_new)

        # Merge points that are too close together
        new_contour = np.column_stack((x_new, y_new))
        final_contour = [new_contour[0]]
        for point in new_contour[1:]:
            if np.linalg.norm(final_contour[-1] - point) >= merge_distance:
                final_contour.append(point)

        return np.array(final_contour, dtype=np.int32).reshape(-1, 1, 2)

    def contour_to_cubic_spline(self, contour, num_points=10000, smoothness=1500, merge_distance=10):
        # Convert contour coordinates to numpy array
        contour = np.array(contour, dtype=np.float32).reshape(-1, 2)

        # 创建参数化变量
        tck, u = splprep([contour[:, 0], contour[:, 1]], s=smoothness)

        # Generate new points on the spline curve using finer intervals
        unew = np.linspace(0, 1, num_points)
        out = splev(unew, tck)

        # Combine new x and y coordinate points
        new_contour = np.column_stack(out)

        # Merge points that are too close together
        final_contour = [new_contour[0]]
        for point in new_contour[1:]:
            if np.linalg.norm(final_contour[-1] - point) >= merge_distance:
                final_contour.append(point)

        # Convert final contours to integer types
        return np.array(final_contour, dtype=np.int32).reshape(-1, 1, 2)

    @staticmethod
    def simplify_contour(contour, threshold):
        simplified_vector = [contour[0]]
        for point in contour[1:]:
            if np.linalg.norm(point - simplified_vector[-1]) >= threshold:
                simplified_vector.append(point)

        return np.array(simplified_vector)

    @staticmethod
    def process_and_blur_contours(contours, image, blur_ksize=17):
        # Create a blank image, the same size as the original image
        blank_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Drawing contours on the blank image
        for contour in contours:
            if isinstance(contour, list):
                # convert contour -> array
                contour = np.array(contour, dtype=np.int32)
            if contour.shape[0] > 0:
                # Reshape contours to ensure correct shape
                reshaped_contour = contour.reshape(-1, 1, 2)
                cv2.drawContours(blank_image, [reshaped_contour], -1, 255, 1)

        # Use Gaussian blur
        blurred_image = cv2.GaussianBlur(blank_image, (blur_ksize, blur_ksize), 0)
        # cv2.imshow('Blurred Image', blurred_image)

        # make it brighter
        adjusted_img = cv2.convertScaleAbs(blurred_image, alpha=5.0, beta=0)
        # cv2.imshow('Adjusted Image', adjusted_img)

        # Apply thinning
        thinned_image = ximgproc.thinning(adjusted_img)
        # cv2.imshow('Thinned Image', thinned_image)

        thinned_contours, _ = cv2.findContours(thinned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return thinned_contours

    def process_contours(self, contours, facial_contours, image):
        # Combine facial feature contours into one list
        facial_feature_points = []
        for face in facial_contours:
            for feature, points in face.items():
                facial_feature_points.extend(points)

        # Convert facial feature points to a NumPy array
        facial_feature_points = np.array(facial_feature_points)

        # Process each contour
        processed_contours = []
        non_facial_contours = []
        for contour in contours:
            if len(contour) > 3:
                # Check if the contour is part of the facial features
                if self.is_contour_in_facial_features(contour, facial_feature_points):
                    # Keep the contour as it is
                    processed_contours.append(contour)
                else:
                    # Simplify the contour
                    try:
                        simp_non_facial_contour = self.simplify_contour(contour, 70)
                        non_facial_contours.append(simp_non_facial_contour)
                    except ValueError as e:
                        # Print error & skip this contour
                        print(f"Error processing contour: {e}")
                        continue

        # Process the non-facial contours collectively, assuming this function returns contours
        if non_facial_contours:
            refined_non_facial_contours = self.process_and_blur_contours(non_facial_contours, image)
            processed_contours.extend(refined_non_facial_contours)  # Extend if multiple contours returned

        return processed_contours

    @staticmethod
    def is_contour_in_facial_features(contour, facial_feature_points):
        # Check if any point in the contour is close to a facial feature point
        for point in contour.reshape(-1, 2):
            if np.min(np.linalg.norm(facial_feature_points - point, axis=1)) < 10:
                return True
        return False

    def img_processing(self, image, method, predictor_path):

        # Detect facial features
        facial_contours = self.sense_detector.detect_senses(predictor_path, image)

        # Flatten the list of facial feature points
        all_points = [point for feature in facial_contours[0].values() for point in feature]

        # Define a region around the facial features
        x_min = min([point[0] for point in all_points])
        x_max = max([point[0] for point in all_points])
        y_min = min([point[1] for point in all_points])
        y_max = max([point[1] for point in all_points])

        # Adjust padding: less horizontal padding, more vertical padding
        horizontal_padding = 100  # Reduce for narrower width
        vertical_padding = 300  # Increase for longer height
        x_min = max(0, x_min - horizontal_padding)
        x_max = min(image.shape[1], x_max + horizontal_padding)
        y_min = max(0, y_min - vertical_padding)
        y_max = min(image.shape[0], y_max + vertical_padding)

        # Crop the image to the region of interest
        ori_facial_region = image[y_min:y_max, x_min:x_max]

        # Remove background
        print("Removing background...")
        ori_image_bkg_removal = self.remove_background(ori_facial_region)
        ori_image_bkg_removal_grey = cv2.cvtColor(ori_image_bkg_removal, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Selfie - colour", ori_facial_region)

        # Resize the img to fixed size 360x480, -> easier to handle the edges/contours
        print("Rescaling the img to 360*480...")
        resized_ori_image_bkg_removal_grey = self.resize_with_aspect_ratio(ori_image_bkg_removal_grey, 360, 480)

        # cv2.imshow("Resized Selfie - bkg removed - grey",
        #            resized_ori_image_bkg_removal_grey)  # Display the resized image

        # NOW detect facial features again & get coordinates
        print("Finding senses...")
        resized_ori_facial_contours = self.sense_detector.detect_senses(predictor_path,
                                                                        resized_ori_image_bkg_removal_grey)

        # Apply Canny edge detector
        print("Blurring img...")
        resized_ori_blurred_img = cv2.GaussianBlur(resized_ori_image_bkg_removal_grey, (3, 3), sigmaX=1, sigmaY=1)

        # Use Canny to find edges
        print("Finding edges & contours...")
        resized_ori_edges = cv2.Canny(resized_ori_blurred_img, threshold1=40,
                                      threshold2=80)  # 50 100 using photos from online | 40 70 using camera
        # cv2.imshow('ori_Edges', resized_ori_edges)
        # Find contours on edges
        contours, _ = cv2.findContours(resized_ori_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process contours
        print("Simplifying contours...")
        processed_contours = self.process_contours(contours, resized_ori_facial_contours,
                                                   resized_ori_edges)  # keep facial contour's details as much as possible

        # 1. Origin img -> bkg removal -> greyscale -> find edge -> find contours
        # 2. Get senses area -> separate contours to 2.1 facial and 2.2 non_facial
        # 3. keep facial's contours detail & blur non_facial contours img
        # 4. Increase non_facial img contrast & use ximgproc.thinning() to get line's 'skeleton'
        # 5. Save the non_facial img & draw the facial's contours on that img

        # Convert contours to a5 size
        print("Converting contours into A5 size...")
        a5_contours = self.convert_contours_to_a5_size(processed_contours, resized_ori_edges.shape[:2])
        a5_image = np.zeros((2480, 1748, 3), dtype=np.uint8)
        # Draw contours on a5 size paper
        for contour in a5_contours:
            cv2.drawContours(a5_image, [contour], -1, [255, 255, 255], 1)
        # self.cv2_imshow_resize_image_for_display('A5 Contours', a5_image)
        # Save the original A5 size vectorized image
        cv2.imwrite('A5 Contours.jpg', a5_image)

        # custom vectorized contours
        print(f"Vectorizing contours by using {method.upper()} method...")
        custom_vectorized_contours = self.vectorize_contour(a5_contours, method)
        a5_drawing_board = np.zeros_like(a5_image)
        for contour in custom_vectorized_contours:
            cv2.drawContours(a5_drawing_board, [contour], -1, [255, 255, 255], 1)
        # self.cv2_imshow_resize_image_for_display('a5_drawing_board', a5_drawing_board)
        cv2.imwrite(f'{method.upper()} vectorized_a5_drawing_board.jpg', a5_drawing_board)

        # update final contours & images
        self.final_contours = custom_vectorized_contours
        self.final_image = a5_drawing_board

        cv2.waitKey(0)
        # # Wait for a key press and close windows if 'q' is pressed
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()

    def update_final_contours(self):
        return self.final_contours
