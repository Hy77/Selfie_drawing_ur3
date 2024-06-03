import numpy as np
from simplification.cutil import simplify_coords
import matplotlib.pyplot as plt
from concorde.tsp import TSPSolver
from mpl_toolkits.mplot3d import Axes3D


class PathPlanner:
    def __init__(self, final_contours):
        self.final_contours = final_contours
        self.nestedcoords = []
        self.sendcoords= []
        self.storedsimcoordinates=[]
        self.listofcoords=[]
        self.invertedy=[]
        self.convertedm=[]

    def simplify(self):
        # print(self.final_contours)
        for contour in self.final_contours:
            contour_points = np.squeeze(contour)
            simplifycoords = simplify_coords(contour_points, 22)

            self.storedsimcoordinates.append(simplifycoords)
        print("storedsimcoordinates",self.storedsimcoordinates)
        print("type storedsimcoordinates:", type(self.storedsimcoordinates))

    def tsp_algo(self):
        self.simplify()
        firstcoords=[]
        lastcoords=[]
        for array in self.storedsimcoordinates:
            firstcoords.append(array[0])
            lastcoords.append(array[-1])
        firstcoords = np.array(firstcoords)
        lastcoords = np.array(lastcoords)
        print("firstcoords",firstcoords)

        solver = TSPSolver.from_data(firstcoords[:,0], lastcoords[:,1], norm='EUC_2D')

        tour_data = solver.solve()

        path = tour_data.tour

        for x in path:
            coords = self.storedsimcoordinates[x]
            self.nestedcoords.append(coords)
        print("nested", self.nestedcoords)

        height = 2480

        for coords in self.nestedcoords:
            coords[:, 1] = height - coords[:, 1]
            self.invertedy.append(coords)
        # print("invertedy",self.invertedy)

        dpi = 300
        dpmm = dpi / 25.4
        dpm = dpmm * 1000
        pixelperm = 1 / dpm
        print("pixelperm:", pixelperm)
        for coordinatess in self.invertedy:
            converttomcoords = coordinatess * pixelperm
            self.convertedm.append(converttomcoords)

    def visualization(self):
        # Run tsp algorithm
        self.tsp_algo()
        sum = 0
        for coords in self.nestedcoords:
            sum += coords.shape[0]
        print(sum)
        for coords in self.nestedcoords:
            # plt.gca().invert_yaxis()
            print(f"coords {coords}:",coords.shape)
            plt.plot(coords[:,0], coords[:,1], '-o')
        plt.show()

        for coordds in self.convertedm:
            # plt.gca().invert_yaxis()
            print(f"coords {coordds}:",coordds.shape)
            plt.plot(coordds[:,0], coordds[:,1], '-o')
        plt.show()

    def scaling(self):
        # self.tsp_algo()
        global_corners = np.array([
            (-0.14819689315343929, 0.05677191725504173),
            (-0.44357801592252305, 0.05677191725504173),
            (-0.44357801592252305, -0.15015018571761504),
            (-0.14819689315343929, -0.15015018571761504)
        ], dtype=np.float32)

        local_corners= np.array([
            (0.148, 0),
            (0.148, 0.210),
            (0, 0.210),
            (0, 0),
        ], dtype=np.float32)
        print(self.nestedcoords)
        # A @ T = B
        T = np.linalg.pinv(local_corners) @ global_corners
        print(T.shape)
        original_length = len(self.convertedm)

        plan_contour = []
        height = 0.026212555279538863
        for contour in self.convertedm[: int(original_length)]:
            contour_stack_z = np.column_stack((contour, np.ones(contour.shape[0])*height))
            duplicate_point = np.array((contour[-1][0], contour[-1][1], height + height))
            contour_z = np.vstack((contour_stack_z, duplicate_point))
            plan_contour.append(contour_z)

        final_contour = plan_contour[0]
        for contour in plan_contour[1:]:
            final_contour = np.vstack((final_contour, contour))

        print(final_contour.shape)
        checker = final_contour
        checker[:,:2] = final_contour[:,:2] @ T
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(checker[:,0], checker[:,1], checker[:,2])
        plt.show()

        print(checker)
        self.listofcoords=checker
        for lists in self.listofcoords:
            listofcoords=tuple(lists)
            self.sendcoords.append(listofcoords)
        print(self.sendcoords)
        print(type(self.sendcoords))

    def update_tsp_coordinates(self):
        print("list of coords",self.sendcoords)
        return self.sendcoords
