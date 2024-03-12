#!/usr/bin/env python3
# img_processing.py

# This will assume the senses' position and use diff threshold to simplify the contour

import cv2
import scipy
import numpy as np
from scipy.interpolate import interp1d, splprep, splev
from PIL import Image
from sense_detector import SenseDetector

class ImgProcessor():

    def __init__(self):
        self.sense_detector = SenseDetector
        self.n_vector = None
        self.b_vector = None
        self.c_vector = None

    # Use can define the img processing method
    def vectorize_contour(self, contour, method='n'):
        if method == 'n':  # nearest neighbor
            return self.contour_to_nearest_neighbor_vector(contour)
        elif method == 'b':  # bilinear
            return self.contour_to_bilinear_vector(contour)
        elif method == 'c':  # bicubic
            return self.contour_to_cubic_spline(contour)
        else:
            raise ValueError(
                "Invalid vectorization method. Choose 'n' for nearest neighbor, 'b' for bilinear, or 'c' for bicubic.")

    @staticmethod
    def remove_background(image):
        # Create a mask the same size as the image and initialize it with 0
        mask = np.zeros(image.shape[:2], np.uint8)

        # Create the foreground and background model required by the grabCut algorithm
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # Define a rectangular area (starting point and end point) that contains the foreground object
        rect = (50, 50, image.shape[1] - 100, image.shape[0] - 100)

        # Apply grabCut algorithm
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        # Converts background and possible background pixels to 0 and others to 1
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # Remove background using mask
        image_bkg_removal = image.copy()
        image_bkg_removal[mask2 == 0] = [255, 255, 255]

        return image_bkg_removal

    @staticmethod
    def resize_and_center_on_a3(image, border_size):
        # A3 size in pixels at 300 dpi
        a3_width, a3_height = 3508, 4961

        # Convert OpenCV image to PIL format
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Get image dimensions
        original_width, original_height = image_pil.size

        # Calculate the new size while maintaining the aspect ratio
        aspect_ratio = original_width / original_height
        if a3_width / a3_height >= aspect_ratio:
            new_width = int(a3_height * aspect_ratio)
            new_height = a3_height
        else:
            new_width = a3_width
            new_height = int(a3_width / aspect_ratio)

        # Resize the image
        resized_image_pil = image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create a new image with a white background
        new_image = Image.new('RGB', (a3_width, a3_height), 'white')

        # Calculate the position to paste the resized image
        x_offset = (a3_width - new_width) // 2
        y_offset = (a3_height - new_height) // 2

        # Paste the resized image onto the center of the new image
        new_image.paste(resized_image_pil, (x_offset, y_offset))

        # Add border
        border_image = Image.new('RGB', (a3_width - 2 * border_size, a3_height - 2 * border_size), 'white')
        border_image.paste(new_image, (border_size, border_size))

        # Convert PIL image back to OpenCV format
        open_cv_image = cv2.cvtColor(np.array(border_image), cv2.COLOR_RGB2BGR)

        return open_cv_image

    @staticmethod
    def resize_image_for_display(image, max_height=800):
        height, width = image.shape[:2]
        if height > max_height:
            scaling_factor = max_height / height
            resized_image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            return resized_image
        return image

    @staticmethod
    def merge_close_lines(contours, max_distance=2):
        # 处理每个轮廓
        for i, contour in enumerate(contours):
            if len(contour) < 2:  # 忽略单点轮廓
                continue
            # 使用 kd-tree 算法来找出相邻的点
            tree = scipy.spatial.KDTree(contour)
            pairs = list()
            # 检查每个点，查找距离过近的点对
            for j, point in enumerate(contour):
                indices = tree.query_ball_point(point, r=max_distance)
                for k in indices:
                    if j < k:  # 避免自身比较以及重复的点对
                        pairs.append((j, k))
            # 合并需要合并的点对
            for j, k in pairs:
                if j < len(contour) and k < len(contour):  # 确保索引有效
                    contour[j] = (contour[j] + contour[k]) / 2
                    contour = np.delete(contour, k, 0)
            contours[i] = contour
        return contours

    @staticmethod
    def adaptive_simplify_vector(contour, image_dim, min_threshold=3, max_threshold=35, focus_area=None):
        # Calculate the proportion of the outline's bounding box size relative to the image size
        _, _, w, h = cv2.boundingRect(contour)
        contour_size_ratio = max(w, h) / max(image_dim)

        # Determine the center of the image
        image_center = np.array(image_dim) / 2

        # Initialize simplified vector with the first point
        simplified_vector = [contour[0]]

        # Calculate the distance of each contour point from the image center
        distances_from_center = np.sqrt(((contour - image_center) ** 2).sum(axis=1))

        # Decide on the threshold based on the position of the contour point
        for point, distance in zip(contour[1:], distances_from_center[1:]):
            print(distance, contour_size_ratio)
            if focus_area is not None and distance <= focus_area:
                threshold = min_threshold
            else:
                threshold = max_threshold
            if np.linalg.norm(np.array(point) - np.array(simplified_vector[-1])) >= threshold:
                simplified_vector.append(point)

        # Ensure the final contour is closed
        if not np.array_equal(simplified_vector[0], simplified_vector[-1]):
            simplified_vector.append(simplified_vector[0])

        return np.array(simplified_vector, dtype=np.int32)

    def contour_to_nearest_neighbor_vector(self, contour):
        # Make sure the outline is a two-dimensional array
        contour = contour.reshape(-1, 2)
        vectorized_contour = [contour[0]]  # Initialising the vectorised profile

        for i in range(1, len(contour)):
            prev_point = vectorized_contour[-1]
            cur_point = contour[i]

            # Check whether the line segments are aligned along the x-axis or the y-axis.
            if abs(prev_point[0] - cur_point[0]) > abs(prev_point[1] - cur_point[1]):
                # x-axis alignment
                new_point = (cur_point[0], prev_point[1])
            else:
                # y-axis alignment
                new_point = (prev_point[0], cur_point[1])

            vectorized_contour.append(new_point)
            vectorized_contour.append(cur_point)

        self.n_vector = np.array(vectorized_contour, dtype=int)

        return self.n_vector

    def contour_to_bilinear_vector(self, contour, num_points=100):
        # Make sure the outline is a two-dimensional array
        contour = contour.reshape(-1, 2)
        x = np.array(contour[:, 0])
        y = np.array(contour[:, 1])

        # Create parameterised variables
        t = np.linspace(0, 1, len(x))
        t_new = np.linspace(0, 1, num_points)

        # Create interpolating functions and generate smooth curves
        f1 = interp1d(t, x, kind='quadratic')
        f2 = interp1d(t, y, kind='quadratic')

        # Use the interpolating function to generate new points
        x_new = f1(t_new)
        y_new = f2(t_new)

        # Form new points into contours
        self.b_vector = np.stack((x_new, y_new), axis=1).astype(int)

        return self.b_vector

    def contour_to_cubic_spline(self, contour, num_points=100):

        self.c_vector = contour
        return self.c_vector

    def img_processing(self, image, method, size):

        # Remove background
        image_bkg_removal = self.remove_background(image)

        # Resize and center on A3
        image_a3 = self.resize_and_center_on_a3(image_bkg_removal, 50)

        # define the threshold for Canny edge detector
        th1 = 100
        th2 = 200
        # check if we proceed with origin img or a3 img
        if size == 'a3':
            processed_image = image_a3
            th1 = 15
            th2 = 40
        else:
            processed_image = image_bkg_removal

        # Convert to greyscale
        greyscale_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detector
        edges = cv2.Canny(greyscale_image, threshold1=th1, threshold2=th2)
        # cv2.imwrite(f'edge_{size}.jpg', edges)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Vectorised contours based on user-selected methods
        vectorized_contours = []
        for contour in contours:
            if len(contour) > 3:
                try:
                    vectorized_contour = self.vectorize_contour(contour, method=method)
                    vectorized_contours.append(vectorized_contour)
                except ValueError as e:
                    # print error & skip this contour
                    print(f"Error processing contour: {e}")
                    continue

        # Simplify vectorized contours
        image_dim = processed_image.shape[:2]  # get img size
        focused_simplified_contours = [self.adaptive_simplify_vector(contour, image_dim, focus_area=150) for contour in
                                       vectorized_contours]
        # 假设contour是从图像中提取的轮廓点集
        focused_simplified_contours = self.merge_close_lines(focused_simplified_contours, 1)

        # Visualisation of simplified vectorised profiles
        vectorized_image = np.zeros_like(processed_image)  # Use the A3 size image
        for contour in focused_simplified_contours:
            # Ensure the contour coordinates are within the image dimensions
            contour = np.clip(contour, 0, np.array(processed_image.shape[:2][::-1]) - 1)
            cv2.polylines(vectorized_image, [contour], isClosed=False, color=(255, 255, 255), thickness=1)

        # Display the images
        # cv2.imshow('Origin', self.resize_image_for_display(image))
        # cv2.imshow('Foreground', self.resize_image_for_display(processed_image))
        # cv2.imshow('Greyscale Image', self.resize_image_for_display(greyscale_image))
        if size == 'a3':
            cv2.imshow('Greyscale Image', self.resize_image_for_display(greyscale_image))
            cv2.imshow('Edges', self.resize_image_for_display(edges))
            cv2.imshow(f'{method.upper()} Vectorized Contours', self.resize_image_for_display(vectorized_image))
        else:
            cv2.imshow('Greyscale Image', greyscale_image)
            cv2.imshow('Edges', edges)
            cv2.imshow(f'{method.upper()} Vectorized Contours', vectorized_image)

        # Save the original A3 size vectorized image
        cv2.imwrite(f'img_results/a3_size_img_results/{method.upper()} vectorized_image_a3.jpg', vectorized_image)

        # Show images & press 'q' to exit
        cv2.waitKey(0)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
