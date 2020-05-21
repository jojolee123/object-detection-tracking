# coding:utf-8
# *********************************************************************************************************
#一个cell为8*8，每个像素点有梯度和方向两个值，将其累加到9个bin中（一个bin为20°），即8*8用9个值来表示，一个block中有4个cell
#即一个block用36维的行向量表示（记得归一化），一般选取128*64的图像，可以有5*17个block，5*17*36=3780，所以衣服图像用3780维行向量表示
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from PIL import Image

# 灰度图像gamma校正
def gamma(img):
    # 不同参数下的gamma校正
    # img1 = img.copy()
    # img2 = img.copy()
    # img1 = np.power( img1 / 255.0, 0.5 )
    # img2 = np.power( img2 / 255.0, 2.2 )
    return np.power(img / 255.0, 1)


# 获取梯度值cell图像，梯度方向cell图像
def div(img, cell_x, cell_y, cell_w):
    cell = np.zeros(shape=(cell_x, cell_y, cell_w, cell_w))
    img_x = np.split(img, cell_x, axis=0)
    for i in range(cell_x):
        img_y = np.split(img_x[i], cell_y, axis=1)
        for j in range(cell_y):
            cell[i][j] = img_y[j]
    return cell


# 获取梯度方向直方图图像，每个像素点有9个值
def get_bins(grad_cell, ang_cell):
    bins = np.zeros(shape=(grad_cell.shape[0], grad_cell.shape[1], 9))
    for i in range(grad_cell.shape[0]):
        for j in range(grad_cell.shape[1]):
            binn = np.zeros(9)
            grad_list = np.int8(grad_cell[i, j].flatten())  # 每个cell中的64个梯度值展平，并转为整数
            ang_list = ang_cell[i, j].flatten()  # 每个cell中的64个梯度方向展平)
            ang_list = np.int8(ang_list / 20.0)  # 0-9
            ang_list[ang_list >= 9] = 0
            for m in range(len(ang_list)):
                binn[ang_list[m]] += int(grad_list[m])  # 不同角度对应的梯度值相加，为直方图的幅值
            # 每个cell的梯度方向直方图可视化
            # N = 9
            # x = np.arange( N )
            # str1 = ( '0-20', '20-40', '40-60', '60-80', '80-100', '100-120', '120-140', '140-160', '160-180' )
            # plt.bar( x, height = binn, width = 0.8, label = 'cell histogram', tick_label = str1 )
            # for a, b in zip(x, binn):
            # plt.text( a, b+0.05, '{}'.format(b), ha = 'center', va = 'bottom', fontsize = 10 )
            # plt.show()
            bins[i][j] = binn
    return bins


# 计算图像HOG特征向量，长度为 15*7*36 = 3780
def hog(img, cell_x, cell_y, cell_w):
    height, width = img.shape
    gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # x方向梯度
    gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # y方向梯度
    gradient_magnitude = np.sqrt(np.power(gradient_values_x, 2) + np.power(gradient_values_y, 2))
    gradient_angle = np.arctan2(gradient_values_x, gradient_values_y)
    # plt.figure()
    # plt.subplot( 1, 2, 1 )
    # plt.imshow(gradient_angle)
    # 角度转换至（0-180）
    gradient_angle[gradient_angle > 0] *= 180 / 3.14
    gradient_angle[gradient_angle < 0] = (gradient_angle[gradient_angle < 0] + 3.14) * 180 / 3.14
    # plt.subplot( 1, 2, 2 )
    # plt.imshow( gradient_angle )
    # plt.show()

    grad_cell = div(gradient_magnitude, cell_x, cell_y, cell_w)
    ang_cell = div(gradient_angle, cell_x, cell_y, cell_w)
    bins = get_bins(grad_cell, ang_cell)
    feature = []
    for i in range(cell_x - 1):
        for j in range(cell_y - 1):
            tmp = []
            tmp.append(bins[i, j])
            tmp.append(bins[i + 1, j])
            tmp.append(bins[i, j + 1])
            tmp.append(bins[i + 1, j + 1])
            mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
            tmp = np.concatenate(tmp, axis=0)
            magnitude = mag(tmp)
            if magnitude != 0:
                normalize = lambda tmp, magnitude: [element / magnitude for element in tmp]
                tmp = normalize(tmp, magnitude)
            feature.append(tmp)


    return np.array(feature).flatten().reshape(1, -1)


if __name__ == '__main__':
    # a = np.load('train.bin', allow_pickle=True)
    # print(a.shape)
    ROOT1 = 'D:\python_code\图像处理\新建文件夹/train/train/neg_person/'
    ROOT2 = 'D:\python_code\图像处理\新建文件夹/train/train/pos_person/'
    flag = 0
    label = 0
    for ROOT in [ROOT1, ROOT2]:
        for name in os.listdir(ROOT):

            img = cv2.imread(ROOT + name, cv2.IMREAD_GRAYSCALE)
            if (img is None):
                print('Not read image.')
            cell_w = 8
            cell_x = int(img.shape[0] / cell_w)  # cell行数
            cell_y = int(img.shape[1] / cell_w)  # cell列数
            gammaimg = gamma(img) * 255
            feature = hog(gammaimg, cell_x, cell_y, cell_w)
            feature = np.hstack([feature, np.array(label).reshape(1, -1)])
            if flag == 0:
                features = feature
                flag = 1
            else:
                features = np.concatenate((features, feature), axis=0)
            # with open('train.bin', 'wb') as f:
            #     feature.tofile(f)
            flag += 1
            if flag % 300 == 0:
                print(flag)
        label += 1

    np.savetxt('train.txt', features, fmt='%f', delimiter=' ')
    np.save("train.npy", features)

