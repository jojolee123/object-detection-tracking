import joblib
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from HOG import gamma, hog
from nms import nms

def load_model(modelpath, fold=0):
    cls = joblib.load(filename=f'{modelpath}')
    return cls

def predict(filepath):

    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (64, 128))
    assert image.shape[0] == 128, "image shape mismatch"
    assert image.shape[1] == 64, "image shape mismatch"
    cell_w = 8
    cell_x = int(image.shape[0] / cell_w)  # cell行数
    cell_y = int(image.shape[1] / cell_w)  # cell列数
    gammaimg = gamma(image) * 255
    feature = hog(gammaimg, cell_x, cell_y, cell_w)
    cls = load_model('model/svm_fold0.pkl')
    pred = cls.predict(feature)
    return pred


def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def detect(img, clf, threshold=.5, visualize_det=False):
    min_wdw_sz = (64, 128)
    step_size = (16, 32)
    im_scaled = img
    cell_w = 8
    cell_x = int(min_wdw_sz[1] / cell_w)  # cell行数
    cell_y = int(min_wdw_sz[0] / cell_w)  # cell列数
    detections = []
    cd = []
    for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
        if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
            continue
        # Calculate the HOG features
        im_window = gamma(im_window)
        fd = hog(im_window, cell_x, cell_y, cell_w)
        pred = clf.predict(fd)
        if pred > 0.8:
            print("Detection:: Location -> ({}, {})".format(x, y))
            print("Scale ->  {} | Confidence Score {} \n".format(1, clf.decision_function(fd)))
            detections.append((x, y, clf.decision_function(fd), int(min_wdw_sz[0]), int(min_wdw_sz[1])))
            cd.append(detections[-1])
        # If visualize is set to true, display the working
        # of the sliding window
        if visualize_det:
            clone = im_scaled.copy()
            for x1, y1, _ in cd:
                # Draw the detections at this scale
                cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 +
                                                im_window.shape[0]), (0, 0, 0), thickness=2)
            cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y +
                                          im_window.shape[0]), (255, 255, 255), thickness=2)
            cv2.imshow("Sliding Window in Progress", clone)
            cv2.waitKey(30)
    clone = img.copy()
    for (x_tl, y_tl, _, w, h) in detections:
        # Draw the detections
        cv2.rectangle(img, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 0), thickness=2)
    cv2.imshow("Raw Detections before NMS", img)
    cv2.waitKey()

    detections = nms(detections, threshold)
    for (x_tl, y_tl, _, w, h) in detections:
        # Draw the detections
        cv2.rectangle(clone, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 0), thickness=2)
    cv2.imshow("Final Detections after applying NMS", clone)
    cv2.waitKey()
if __name__ == "__main__":
    # pred = predict('image/1.jpg')
    img = cv2.imread('image/1.jpg', cv2.IMREAD_GRAYSCALE)
    modelpath = 'model/svm_fold0.pkl'
    clf = load_model(modelpath)
    detect(img, clf, visualize_det=False)