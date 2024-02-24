#!/usr/bin/env python3
import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.interpolate import interp1d


# Use can define the img processing method
def vectorize_contour(contour, method='n'):
    if method == 'n':  # 最近邻向量化
        return contour_to_nearest_neighbor_vector(contour)
    elif method == 'b':  # 双线性向量化
        return contour_to_bilinear_vector(contour)
    elif method == 'c':  # 双三次向量化（暂未实现）
        return contour  # 返回原始轮廓作为占位符
    else:
        raise ValueError("Invalid vectorization method. Choose 'n' for nearest neighbor, 'b' for bilinear, or 'c' for bicubic.")


def contour_to_nearest_neighbor_vector(contour):
    # 确保轮廓是一个二维数组
    contour = contour.reshape(-1, 2)
    vectorized_contour = [contour[0]]  # 初始化向量化轮廓

    for i in range(1, len(contour)):
        prev_point = vectorized_contour[-1]
        cur_point = contour[i]

        # 检查线段是沿 x 轴对齐还是沿 y 轴对齐
        if abs(prev_point[0] - cur_point[0]) > abs(prev_point[1] - cur_point[1]):
            # x 轴对齐
            new_point = (cur_point[0], prev_point[1])
        else:
            # y 轴对齐
            new_point = (prev_point[0], cur_point[1])

        vectorized_contour.append(new_point)
        vectorized_contour.append(cur_point)

    return np.array(vectorized_contour, dtype=int)


def contour_to_bilinear_vector(contour, num_points=100):
    # 确保轮廓是一个二维数组
    contour = contour.reshape(-1, 2)
    x = np.array(contour[:, 0])
    y = np.array(contour[:, 1])

    # 创建参数化变量
    t = np.linspace(0, 1, len(x))
    t_new = np.linspace(0, 1, num_points)

    # 创建插值函数并生成平滑的曲线
    f1 = interp1d(t, x, kind='quadratic')
    f2 = interp1d(t, y, kind='quadratic')

    # 使用插值函数生成新的点
    x_new = f1(t_new)
    y_new = f2(t_new)

    # 将新的点组成轮廓
    smooth_contour = np.stack((x_new, y_new), axis=1).astype(int)
    return smooth_contour


def img_processing(method='n'):
    # Load the image
    image = cv2.imread('photo_1.jpg')

    # Show origin img
    cv2.imshow('Origin', image)

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
    image_bkg_removal = image * mask2[:, :, np.newaxis]

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

    # 根据用户选择的方法向量化轮廓
    vectorized_contours = [vectorize_contour(contour, method=method) for contour in contours if len(contour) > 3]

    # 可视化向量化轮廓
    vectorized_image = np.zeros_like(image)
    for vectorized_contour in vectorized_contours:
        cv2.polylines(vectorized_image, [vectorized_contour], isClosed=False, color=(255, 0, 0), thickness=2)

    cv2.imshow(f'{method.upper()} Vectorized Contours', vectorized_image)

    # Show images & press 'q' to exit
    cv2.waitKey(0)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    method = input("Choose a vectorization method ('n' for nearest neighbor, 'b' for bilinear, or 'c' for bicubic): ").lower()
    img_processing(method)
