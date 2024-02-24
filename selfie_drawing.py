#!/usr/bin/env python3
import cv2
import numpy as np
from scipy.interpolate import splprep, splev


def is_nonlinear(contour):
    # 检查轮廓是否至少包含一个非线性段
    for i in range(len(contour) - 2):
        if not np.allclose(contour[i], 2 * contour[i + 1] - contour[i + 2]):
            return True
    return False


def resample_contour(contour, num_points=25):
    # 确保轮廓是一个二维数组
    contour = contour.reshape(-1, 2)
    # 计算轮廓的弧长
    arc_length = cv2.arcLength(contour, closed=True)
    # 计算每个点之间的间距
    step_size = arc_length / num_points
    # 使用间距重新采样轮廓
    return cv2.approxPolyDP(contour, epsilon=step_size, closed=True)


def contour_to_bezier(contour, smoothing=2.0, num_points=100):
    # 将轮廓点转换为参数化的贝塞尔曲线
    contour = contour.reshape(-1, 2)
    tck, u = splprep(contour.T, u=None, s=smoothing, per=1)
    # 生成曲线上的点
    u_new = np.linspace(u.min(), u.max(), num_points)
    x_new, y_new = splev(u_new, tck, der=0)
    # 将点转换为整数坐标
    points = np.vstack((x_new, y_new)).T.astype(int)
    return points


def img_processing():
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

    # 重新采样并转换轮廓为贝塞尔曲线
    bezier_contours = []
    for contour in contours:
        if len(contour) >= 4:  # 确保轮廓足够长
            resampled_contour = resample_contour(contour, num_points=100)  # 增加重新采样点的数量
            if is_nonlinear(resampled_contour) and len(resampled_contour) >= 4:  # 再次检查轮廓长度
                bezier_contour = contour_to_bezier(resampled_contour, smoothing=2.0)  # 调整平滑参数
                bezier_contours.append(bezier_contour)

    # 可视化贝塞尔曲线
    bezier_image = np.zeros_like(image)
    for bezier_contour in bezier_contours:
        cv2.polylines(bezier_image, [bezier_contour], isClosed=False, color=(255, 0, 0), thickness=2)

    cv2.imshow('Bezier Curves', bezier_image)

    # Show images & press 'q' to exit
    cv2.waitKey(0)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    img_processing()
