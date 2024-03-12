#!/usr/bin/env python3
# img_processing.py

import cv2
import scipy
import numpy as np
from scipy.interpolate import interp1d, splprep, splev
from PIL import Image
from sense_detector import SenseDetector

class ImgProcessor():

    def __init__(self):
        self.sense_detector = SenseDetector()
        self.n_vector = None
        self.b_vector = None
        self.c_vector = None

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

    def merge_nearby_lines(self, simplified_contour, delta_slope=0.01, delta_intercept=5):
        if len(simplified_contour) < 3:
            return simplified_contour.reshape(-1, 2)

        # Convert contour to a list of lines
        lines = [simplified_contour[i:i + 2] for i in range(len(simplified_contour) - 1)]

        # Group lines by their slope and intercept
        groups = []

        for line in lines:
            p1, p2 = line.reshape(4)
            slope, intercept = self.calculate_slope_intercept(p1, p2)

            # Find an existing group for the line or create a new one
            found = False
            for group in groups:
                if self.is_within_threshold(group, slope, intercept, delta_slope, delta_intercept):
                    group['lines'].append(line)
                    group['slopes'].append(slope)
                    group['intercepts'].append(intercept)
                    found = True
                    break

            if not found:
                groups.append({
                    'lines': [line],
                    'slopes': [slope],
                    'intercepts': [intercept]
                })

        # Merge lines in each group to create new lines
        merged_lines = self.merge_line_groups(groups)
        return merged_lines

    @staticmethod
    def calculate_slope_intercept(p1, p2):
        x1, y1, x2, y2 = p1, p2
        if x2 - x1 == 0:  # Avoid division by zero for vertical lines
            slope = float('inf')
            intercept = x1
        else:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
        return slope, intercept

    def is_within_threshold(group, slope, intercept, delta_slope, delta_intercept):
        mean_slope = np.mean(group['slopes'])
        mean_intercept = np.mean(group['intercepts'])
        return (abs(mean_slope - slope) < delta_slope and abs(mean_intercept - intercept) < delta_intercept)

    @staticmethod
    def merge_line_groups(groups):
        merged_lines = []
        for group in groups:
            if len(group['lines']) > 1:  # Only merge if there are at least 2 lines in the group
                # Extract all the start and end points of the lines
                all_points = np.vstack(group['lines']).reshape(-1, 2)
                # Find the average start and end points
                avg_start = np.mean(all_points[::2], axis=0)
                avg_end = np.mean(all_points[1::2], axis=0)
                # Create a new line from the average start and end points
                merged_lines.append(np.vstack([avg_start, avg_end]).astype(int))
            else:
                # If there's only one line in the group, keep it as it is
                merged_lines.append(group['lines'][0])

        # Convert list of lines back into a contour array
        merged_contour = np.vstack(merged_lines).reshape(-1, 1, 2)
        return merged_contour

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
            if len(contour) > 3:
                # Check if the contour is part of the facial features
                if self.is_contour_in_facial_features(contour, facial_feature_points):
                    # Keep the contour as it is
                    processed_contours.append(contour)
                else:
                    # Simplify the contour
                    try:
                        simplified_contour = cv2.approxPolyDP(contour, 10, True)
                        processed_contours.extend(simplified_contour)
                        ''' Find somehow to merge the similar lines
                        # Ensure we have enough points to form lines
                        if len(simplified_contour) > 1:
                            print(len(simplified_contour))
                            merged_simplified_contour = self.merge_nearby_lines(simplified_contour)
                            processed_contours.extend(merged_simplified_contour)
                        else:
                            # If not enough points, skip this contour
                            continue
                        '''
                    except ValueError as e:
                        # Print error & skip this contour
                        print(f"Error processing contour: {e}")
                        continue
        return processed_contours

    @staticmethod
    def is_contour_in_facial_features(contour, facial_feature_points):
        # Check if any point in the contour is close to a facial feature point
        for point in contour.reshape(-1, 2):
            if np.min(np.linalg.norm(facial_feature_points - point, axis=1)) < 5:
                return True
        return False

    def img_processing(self, image, method, size, predictor_path):

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

        # Visualisation of simplified vectorised profiles
        vectorized_image = np.zeros_like(image)
        for contour in custom_vectorized_contours:
            # Ensure the contour coordinates are within the image dimensions
            contour = np.clip(contour, 0, np.array(image.shape[:2][::-1]) - 1)
            cv2.polylines(vectorized_image, [contour], isClosed=False, color=(255, 255, 255), thickness=1)

        # Display the images
        # cv2.imshow('Origin', self.resize_image_for_display(image))
        # cv2.imshow('Foreground', self.resize_image_for_display(processed_image))
        # cv2.imshow('Greyscale Image', self.resize_image_for_display(greyscale_image))
        cv2.imshow('Greyscale Image', greyscale_image)
        cv2.imshow('Edges', edges)
        cv2.imshow(f'{method.upper()} Vectorized Contours', vectorized_image)

        # Save the original A3 size vectorized image
        cv2.imwrite(f'img_results/a3_size_img_results/{method.upper()} vectorized_image_a3.jpg', vectorized_image)

        # Show images & press 'q' to exit
        cv2.waitKey(0)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
