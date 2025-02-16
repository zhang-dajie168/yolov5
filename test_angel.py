import cv2
import numpy as np


def draw_normal_vector(contours, image):
    # 遍历每个轮廓
    for contour in contours:
        # cls = contour[0]
        # segment = contour[1:].reshape(int(len(contour) / 2), 1, 2)
        # cls = contour[0]
        # segment = contour[1:].reshape(int(len(contour) / 2), 1, 2)
        rect = cv2.minAreaRect(contour)
        #     # print(segment)
        # 获取最小外接矩形
        # rect = cv2.minAreaRect(segment)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 绘制矩形框
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

        # 矩形的中心点
        center = (int(rect[0][0]), int(rect[0][1]))

        # 矩形的宽和高
        width = rect[1][0]
        height = rect[1][1]

        # 计算角度（以弧度为单位）
        angle = np.deg2rad(rect[2])

        # 计算法向量（垂直于长边的向量）
        if width > height:
            normal_vector = (-np.sin(angle), np.cos(angle))
        else:
            normal_vector = (np.cos(angle), np.sin(angle))

        # 法向量的终点
        endpoint = (int(center[0] + normal_vector[0] * 50), int(center[1] + normal_vector[1] * 50))
        # 计算法向量与水平方向的角度（以弧度为单位）
        horizontal_vector = (1, 0)  # 水平方向的单位向量
        angle_with_horizontal = np.arccos(np.dot(normal_vector, horizontal_vector))

        # 将角度转换为度数
        angle_with_horizontal_deg = np.rad2deg(angle_with_horizontal)

        # 绘制法向量
        cv2.arrowedLine(image, center, endpoint, (255, 0, 0), 2)

        # 绘制水平线
        # 绘制水平向量（从中心点水平向右）
        vector_length = 100  # 向量长度，可以根据需要调整
        start_point = center
        end_point = (center[0] + vector_length, center[1])
        cv2.arrowedLine(image, start_point, end_point, (255, 0, 0), 2, tipLength=0.1)

        return image, angle_with_horizontal_deg



if __name__ == '__main__':
    image=cv2.imread("/home/ymt/项目文件/SL008-YJ001-PFJC/yolov5/img_2.png")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 查找轮廓
    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour=min(contours, key=cv2.contourArea)
    # 计算每个轮廓的最小矩形
    rect = cv2.minAreaRect(contour)
    #     # print(segment)
    # 获取最小外接矩形
    # rect = cv2.minAreaRect(segment)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # 绘制矩形框
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

    # 矩形的中心点
    center = (int(rect[0][0]), int(rect[0][1]))

    # 矩形的宽和高
    width = rect[1][0]
    height = rect[1][1]

    # 计算角度（以弧度为单位）
    angle = np.deg2rad(rect[2])

    # 计算法向量（垂直于长边的向量）
    if width > height:
        normal_vector = (-np.sin(angle), np.cos(angle))
    else:
        normal_vector = (np.cos(angle), np.sin(angle))

    # 法向量的终点
    endpoint = (int(center[0] + normal_vector[0] * 50), int(center[1] + normal_vector[1] * 50))
    # 计算法向量与水平方向的角度（以弧度为单位）
    horizontal_vector = (1, 0)  # 水平方向的单位向量
    angle_with_horizontal = np.arccos(np.dot(normal_vector, horizontal_vector))

    # 将角度转换为度数
    angle_with_horizontal_deg = np.rad2deg(angle_with_horizontal)

    # 绘制法向量
    cv2.arrowedLine(image, center, endpoint, (255, 0, 0), 2)

    # 绘制水平线
    # 绘制水平向量（从中心点水平向右）
    vector_length = 100  # 向量长度，可以根据需要调整
    start_point = center
    end_point = (center[0] + vector_length, center[1])
    cv2.arrowedLine(image, start_point, end_point, (255, 0, 0), 2, tipLength=0.1)
    # image, angle_with_horizontal_deg=draw_normal_vector(contour, image)
        # min_rect = cv2.minAreaRect(contour)
        # min_rectangles.append(min_rect)
        #



    # 显示结果图像
    cv2.imshow("Result Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()