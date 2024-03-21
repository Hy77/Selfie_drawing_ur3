#!/usr/bin/env python3
# img_processing.py

import cv2
import numpy as np
from scipy.interpolate import interp1d, splprep, splev
from PIL import Image
from sense_detector import SenseDetector
import cv2.ximgproc as ximgproc


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
        # Convert the contour to a NumPy array
        contour = np.array(contour)

        # Ensure the contour has at least three points for interpolation
        if len(contour) < 3:
            return contour

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

    @staticmethod
    def simplify_contour(contour, threshold):
        simplified_vector = [contour[0]]
        for point in contour[1:]:
            if np.linalg.norm(point - simplified_vector[-1]) >= threshold:
                simplified_vector.append(point)

        return np.array(simplified_vector)

    def process_and_blur_contours(self, contours, blur_ksize=17):
        # Create a blank image, the same size as the original image
        blank_image = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)

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
        adjusted_img = cv2.convertScaleAbs(blurred_image, alpha=5.0, beta=0)
        # cv2.imshow('Adjusted Image', adjusted_img)
        thinned_image = ximgproc.thinning(adjusted_img)
        # cv2.imshow('Thinned Image', thinned_image)

        return thinned_image

    def process_contours(self, contours, facial_contours, method):
        # Combine facial feature contours into one list
        facial_feature_points = []
        for face in facial_contours:
            for feature, points in face.items():
                facial_feature_points.extend(points)

        # Convert facial feature points to a NumPy array
        facial_feature_points = np.array(facial_feature_points)

        # Process each contour
        processed_contours = []
        for contour in contours:
            # print(f"Contour shape: {contour.shape}")
            # print(f"Contour:\n{contour}")
            if len(contour) > 3:
                # Check if the contour is part of the facial features
                if self.is_contour_in_facial_features(contour, facial_feature_points):
                    # Keep the contour as it is
                    processed_contours.append(contour)
                else:
                    # Simplify the contour
                    try:
                        simp_contours = self.simplify_contour(contour, 30)
                        # blured_contours = self.process_and_blur_contours(simp_contours, 30)
                        self.non_facial_contours.append(simp_contours)
                    except ValueError as e:
                        # Print error & skip this contour
                        print(f"Error processing contour: {e}")
                        continue

        return processed_contours

    @staticmethod
    def is_contour_in_facial_features(contour, facial_feature_points):
        # Check if any point in the contour is close to a facial feature point
        for point in contour.reshape(-1, 2):
            if np.min(np.linalg.norm(facial_feature_points - point, axis=1)) < 3:
                return True
        return False

    def img_processing(self, image, method, size, predictor_path):

        # restore img info
        self.image = image
        drawing_img = np.zeros_like(image)

        # Remove background
        image_bkg_removal = self.remove_background(image)

        # Convert to greyscale
        greyscale_image = cv2.cvtColor(image_bkg_removal, cv2.COLOR_BGR2GRAY)

        # Detect facial features
        facial_contours = self.sense_detector.detect_senses(predictor_path, greyscale_image)

        # Apply Canny edge detector
        edges = cv2.Canny(greyscale_image, threshold1=100, threshold2=200)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process contours
        processed_contours = self.process_contours(contours, facial_contours, method)

        # custom vectorized contours
        custom_vectorized_contours = self.vectorize_contour(processed_contours, method)

        '''
        1. Origin img -> bkg removal -> greyscale -> find edge -> find contours
        2. Get senses area -> separate contours to 2.1 facial and 2.2 non_facial
        3. keep facial's contours detail & blur non_facial contours img
        4. Increase non_facial img contrast & use ximgproc.thinning() to get line's 'skeleton'
        5. Save the non_facial img & draw the facial's contours on that img
        '''
        # Use non_facial img as the base
        non_facial_image = self.process_and_blur_contours(self.non_facial_contours)
        non_facial_contours, _ = cv2.findContours(non_facial_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Concatenate the two lists
        custom_vectorized_contours.extend(non_facial_contours)
        self.final_contours = custom_vectorized_contours

        for contour in self.final_contours:
            # Ensure the contour coordinates are within the image dimensions
            contour = np.clip(contour, 0, np.array(image.shape[:2][::-1]) - 1)
            cv2.polylines(drawing_img, [contour], isClosed=False, color=(255, 255, 255), thickness=1)
        self.final_image = drawing_img
        # Display the images
        # cv2.imshow('Origin', self.resize_image_for_display(image))
        # cv2.imshow('Foreground', self.resize_image_for_display(processed_image))
        # cv2.imshow('Greyscale Image', self.resize_image_for_display(greyscale_image))
        cv2.imshow('Greyscale Image', greyscale_image)
        cv2.imshow('Edges', edges)
        cv2.imshow(f'{method.upper()} Vectorized Contours', drawing_img)

        # Save the original A3 size vectorized image
        # cv2.imwrite(f'img_results/a3_size_img_results/{method.upper()} vectorized_image_a3.jpg', non_facial_image)

        # Show images & press 'q' to exit
        # cv2.waitKey(0)

    def update_final_contours(self):
        return self.final_contours
