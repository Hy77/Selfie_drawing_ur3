#!/usr/bin/env python3
import cv2
import numpy as np
from scipy.interpolate import interp1d, splprep, splev
from PIL import Image


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
    border_image = Image.new('RGB', (a3_width + 2 * border_size, a3_height + 2 * border_size), 'white')
    border_image.paste(new_image, (border_size, border_size))

    # Convert PIL image back to OpenCV format
    open_cv_image = cv2.cvtColor(np.array(border_image), cv2.COLOR_RGB2BGR)

    return open_cv_image


def resize_image_for_display(image, max_height=800):
    height, width = image.shape[:2]
    if height > max_height:
        scaling_factor = max_height / height
        resized_image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        return resized_image
    return image


# Use can define the img processing method
def vectorize_contour(contour, method='n'):
    if method == 'n':  # nearest neighbor
        return contour_to_nearest_neighbor_vector(contour)
    elif method == 'b':  # bilinear
        return contour_to_bilinear_vector(contour)
    elif method == 'c':  # bicubic
        return contour_to_cubic_spline(contour)
    else:
        raise ValueError("Invalid vectorization method. Choose 'n' for nearest neighbor, 'b' for bilinear, or 'c' for bicubic.")


def adaptive_simplify_vector(contour, image_dim, min_threshold=5, max_threshold=10):
    # Calculate the proportion of the outline's bounding box size relative to the image size
    _, _, w, h = cv2.boundingRect(contour)
    contour_size_ratio = max(w, h) / max(image_dim)

    # Adaptive threshold setting based on contour size
    if contour_size_ratio < 1:
        threshold = min_threshold
    else:  # If the profile is large, a larger threshold can be used
        threshold = max_threshold

    # use simplified vector
    simplified_vector = [contour[0]]  # init vector
    for point in contour[1:]:
        if np.linalg.norm(np.array(point) - np.array(simplified_vector[-1])) >= threshold:
            simplified_vector.append(point)

    return np.array(simplified_vector, dtype=int)


def contour_to_nearest_neighbor_vector(contour):
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

    return np.array(vectorized_contour, dtype=int)


def contour_to_bilinear_vector(contour, num_points=100):
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
    smooth_contour = np.stack((x_new, y_new), axis=1).astype(int)
    return smooth_contour


def contour_to_cubic_spline(contour, num_points=100):

    return contour


def img_processing(method='n'):
    # Load the image
    image = cv2.imread('photo_1.jpg')

    # Show origin img
    cv2.imshow('Origin', image)

    # Remove background
    image_bkg_removal = remove_background(image)

    # Resize and center on A3
    image_a3 = resize_and_center_on_a3(image_bkg_removal, 50)

    # Scale the A3 size img for display
    image_bkg_removal = resize_image_for_display(image_a3)

    # Display the image
    cv2.imshow('Foreground', image_bkg_removal)

    # Convert to greyscale
    greyscale_image = cv2.cvtColor(image_bkg_removal, cv2.COLOR_BGR2GRAY)

    # Display the image
    cv2.imshow('Greyscale Image', greyscale_image)

    # Apply Canny edge detector
    edges = cv2.Canny(greyscale_image, threshold1=100, threshold2=200)

    # Display the edges
    cv2.imshow('Edges', edges)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Vectorised contours based on user-selected methods
    vectorized_contours = []
    for contour in contours:
        if len(contour) > 3:
            try:
                vectorized_contour = vectorize_contour(contour, method=method)
                vectorized_contours.append(vectorized_contour)
            except ValueError as e:
                # print error & skip this contour
                print(f"Error processing contour: {e}")
                continue

    # Simplify vectorized contours
    image_dim = image.shape[:2]  # get img size
    simplified_contours = [adaptive_simplify_vector(contour, image_dim) for contour in vectorized_contours]
    print(len(vectorized_contours[0]), len(simplified_contours[0]))

    # Visualisation of simplified vectorised profiles
    vectorized_image = np.zeros_like(image_bkg_removal)  # Use the same size as the displayed image
    for contour in simplified_contours:
        # Ensure the contour coordinates are within the image dimensions
        contour = np.clip(contour, 0, np.array(image_bkg_removal.shape[:2][::-1]) - 1)
        cv2.polylines(vectorized_image, [contour], isClosed=False, color=(255, 0, 0), thickness=1)

    cv2.imshow(f'{method.upper()} Vectorized Contours', vectorized_image)

    # Show images & press 'q' to exit
    cv2.waitKey(0)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    method = input("Choose a vectorization method ('n' for nearest neighbor, 'b' for bilinear, or 'c' for bicubic): ").lower()
    img_processing(method)
