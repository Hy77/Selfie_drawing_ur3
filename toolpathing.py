import numpy as np
from simplification.cutil import simplify_coords
import matplotlib.pyplot as plt
from concorde.tsp import TSPSolver
from mpl_toolkits.mplot3d import Axes3D


class PathPlanner:
    def __init__(self, final_contours, paper_coord):
        self.final_contours = final_contours
        self.nestedcoords = []
        self.sendcoords = []
        self.storedsimcoordinates = []
        self.listofcoords = []
        self.convertedm = []
        self.transformed_coords=[]
        self.paper_coord = paper_coord

    def simplify(self):
        for contour in self.final_contours:
            contour_points = np.squeeze(contour)
            simplifycoords = simplify_coords(contour_points, 22)
            self.storedsimcoordinates.append(simplifycoords)
        print("storedsimcoordinates", self.storedsimcoordinates)
        print("type storedsimcoordinates:", type(self.storedsimcoordinates))

    def tsp_algo(self):
        self.simplify()
        firstcoords = []
        lastcoords = []
        for array in self.storedsimcoordinates:
            firstcoords.append(array[0])
            lastcoords.append(array[-1])
        firstcoords = np.array(firstcoords)
        lastcoords = np.array(lastcoords)
        print("firstcoords", firstcoords)

        solver = TSPSolver.from_data(firstcoords[:, 0], lastcoords[:, 1], norm='EUC_2D')

        tour_data = solver.solve()

        path = tour_data.tour

        for x in path:
            coords = self.storedsimcoordinates[x]
            self.nestedcoords.append(coords)
        print("nested", self.nestedcoords)

        dpi = 300
        dpmm = dpi / 25.4
        dpm = dpmm * 1000
        pixelperm = 1 / dpm
        print("pixelperm:", pixelperm)
        for coordinatess in self.nestedcoords:
            converttomcoords = coordinatess * pixelperm
            self.convertedm.append(converttomcoords)

    def transform_coordinates(self, global_corners, local_corners):
        # Centering and scaling transformation
        global_center = global_corners.mean(axis=0)
        local_center = local_corners.mean(axis=0)

        scale_factor = 1.2
        scaling_matrix = np.array([
            [0, scale_factor, 0],
            [-scale_factor, 0, 0],
            [0, 0, 1]
        ])

        transform_matrix = np.eye(3)
        transform_matrix[0:2, 0:2] = scaling_matrix[0:2, 0:2]
        transform_matrix[0:2, 2] = global_center - scaling_matrix[0:2, 0:2] @ local_center

        transformed_coords = []
        for coords in self.convertedm:
            coords_homogeneous = np.hstack([coords, np.ones((coords.shape[0], 1))])
            transformed = coords_homogeneous @ transform_matrix.T
            transformed_coords.append(transformed[:, :2])
        return transformed_coords

    def visualization(self):
        self.tsp_algo()

        # # for debugging
        # global_corners = np.array([
        #     (-0.14819689315343929, 0.05677191725504173),
        #     (-0.44357801592252305, 0.05677191725504173),
        #     (-0.44357801592252305, -0.15015018571761504),
        #     (-0.14819689315343929, -0.15015018571761504)
        # ], dtype=np.float32)

        global_corners = self.paper_coord

        local_corners = np.array([
            (0, 0),
            (0.148, 0),
            (0.148, 0.210),
            (0, 0.210)
        ], dtype=np.float32)

        transformed_coords = self.transform_coordinates(global_corners, local_corners)

        sum_points = 0
        for coords in transformed_coords:
            sum_points += coords.shape[0]
        print(sum_points)

        # comment these out if errors pop up
        # for coords in transformed_coords:
        #     print(f"coords {coords}:", coords.shape)
        #     plt.plot(coords[:, 0], coords[:, 1], '-o')
        #
        # plt.xlim(-0.443, -0.148)
        # plt.ylim(-0.15, 0.056)
        # plt.gca().invert_yaxis()
        # plt.show()
        # comment these out if errors pop up

        self.transformed_coords=transformed_coords

    def scaling(self):

        original_length = len(self.transformed_coords)

        plan_contour = []
        height = 0.026212555279538863
        for contour in self.transformed_coords[: int(original_length)]:
            contour_stack_z = np.column_stack((contour, np.ones(contour.shape[0]) * height))
            duplicate_point = np.array((contour[-1][0], contour[-1][1], height + height))
            contour_z = np.vstack((contour_stack_z, duplicate_point))
            plan_contour.append(contour_z)

        final_contour = plan_contour[0]
        for contour in plan_contour[1:]:
            final_contour = np.vstack((final_contour, contour))

        print(final_contour.shape)
        checker = final_contour
        checker[:, :2] = final_contour[:, :2]

        # comment these out if errors pop up
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #
        # ax.plot(checker[:, 0], checker[:, 1], checker[:, 2])
        # plt.show()
        # comment these out if errors pop up

        print(checker)
        self.listofcoords = checker
        for lists in self.listofcoords:
            listofcoords = tuple(lists)
            self.sendcoords.append(listofcoords)
        print(self.sendcoords)
        print(type(self.sendcoords))

    def update_tsp_coordinates(self):
        print("list of coords", self.sendcoords)
        return self.sendcoords
