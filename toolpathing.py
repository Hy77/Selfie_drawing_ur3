import cv2
import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment


class PathPlanner:
    def __init__(self, final_contours, image):
        self.path = None
        self.final_contours = final_contours
        self.image = image

    def tsp_algo(self):

        with open('tspcoordinates.txt', 'w') as coordi:  # Writing data to the txt file as a variable coordi
            tsp_coordinates = []  # Storing all the tsp coordinates
            last_point = None  # Storing the last point of the last contour drawn

        for xx in range(len(self.final_contours)):
            min_distance = np.inf  # Setting to infinity to find small values
            nearest_index_contour = None  # If no contours found, keep as None

        for i, contour in enumerate(self.final_contours):  # Execute the loop for each contour individually

            contour_points = np.squeeze(contour)  # Extracting the coordinates of the contour

            if last_point is not None:  # Makes sure that there is distance
                distance = np.linalg.norm(contour_points[0] - last_point)  # Distance calculated to the last
                # point
            else:
                distance = 0

            if distance < min_distance:  # Making the minimum distance as the distance
                min_distance = distance
                nearest_index_contour = i

        contour = self.final_contours.pop(nearest_index_contour)

        contour_points = np.squeeze(contour)  # Extracting the coordinates of the selected contour

        depth_value = 100  # Constant depth value of 100 millimeters

        dist_matrix = distance_matrix(contour_points, contour_points)  # Calculating the pairwise distance
        # matrix. contour_points are the points that gives distance between every point in the contour

        row, col = linear_sum_assignment(dist_matrix)  # Solve the TSP using the Hungarian Algorithm.
        # linear_sum_assignment uses the Hungarian Algorithm which finds the best row and column for reducing
        # the distance travelled

        ordered_contour_points = contour_points[col]  # Reordering the contour points of the TSP

        # Converting the 2D TSP coordinates to 3D TSP coordinates that has a constant depth value
        tsp_coordinates_3d = np.hstack(
            (ordered_contour_points, np.full((len(ordered_contour_points), 1), depth_value)))

        tsp_coordinates.append(tsp_coordinates_3d)  # Adding the 3d coordinates to the tsp coordinates

        last_point = tsp_coordinates_3d[-1, :2]  # Last points are being updated which are x and y in the
        # last element

        # Writing the TSP coordinates to the txt file
        for coordinate in tsp_coordinates_3d:
            coordi.write(f"{coordinate[0]}, {coordinate[1]},{coordinate[2]}\n")

    # Visualization
    animate = np.zeros_like(self.image)
    # Drawing the curves
    for tsp_coords in tsp_coordinates:  # Looping each coordinates
        for i in range(len(tsp_coords) - 1):  # Loops from 0 to the length of coordinates
            starting = (int(tsp_coords[i][0]), int(tsp_coords[i][1]))  # Starting point of the tsp
            # coordinates. i is the point of the current tsp coordinates
            ending = (int(tsp_coords[i + 1][0]), int(tsp_coords[i + 1][1]))  # Ending point of the tsp
            # coordinates.
            cv2.line(animate, starting, ending, (0, 255, 0), 2)  # Draws lines of thickness of
            # green color
            cv2.imshow("Animate", animate)  # Showing the picture
            cv2.waitKey(10)  # Waiting for a period of time to see the drawn lines per iteration

    # Displaying the image with the coordinate points
    cv2.imshow('TSP Coordinates', self.image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
