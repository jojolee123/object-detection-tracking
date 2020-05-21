import cv2 as cv
import sys
import numpy as np
from KCFTracker import KCFTracker
from MOSSETracker import MOSSETracker

def imcv2_recolor(im, a=.1):
    # t = [np.random.uniform()]
    # t += [np.random.uniform()]
    # t += [np.random.uniform()]
    # t = np.array(t) * 2. - 1.
    t = np.random.uniform(-1, 1, 3)

    # random amplify each channel
    im = im.astype(np.float)
    im *= (1 + t * a)
    mx = 255. * (1 + a)
    up = np.random.uniform(-1, 1)
    im = np.power(im / mx, 1. + up * .5)
    # return np.array(im * 255., np.uint8)
    return im

class object_detector:

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.framework = None
        self.load_model()

    def load_model(self):
        if self.model.endswith('weights') and self.cfg.endswith('cfg'):
            self.net = cv.dnn.readNetFromDarknet(self.cfg, self.model)
            self.framework = 'Darknet'
        elif self.model.endswith('caffemodel') and self.cfg.endswith('prototxt'):
            self.net = cv.dnn.readNetFromCaffe(self.cfg, self.model)
            self.framework = 'Caffe'
        else:
            sys.exit('Wrong input for model weights and cfg')

        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    def predict(self, frame):

        # Create a 4D blob from a frame.
        if self.framework == 'Darknet':
            # blob = cv.dnn.blobFromImage(frame, 0.007843, (416, 416), 127.5, crop = False)
            blob = cv.dnn.blobFromImage(cv.resize(frame, (416, 416)), 0.003921, (416, 416), (0, 0, 0), swapRB=True,
                                        crop=False)
        else:
            blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        # Run a model
        self.net.setInput(blob)
        out = self.net.forward()

        return out


def detect(stream, predictor, threshold, classes, track_method):
    _, frame = stream.read()
    predictions = predictor.predict(frame)
    objects_detected = postprocess(frame, predictions, threshold, classes, predictor.framework)

    objects_list = list(objects_detected.keys())
    print('Tracking the following objects', objects_list)
    trackers_dict = dict()
    # multi_tracker = cv.MultiTracker_create()

    if len(objects_list) > 0:

        # trackers_dict = {key: cv.TrackerKCF_create() for key in objects_list}
        if track_method == "KCF":
            trackers_dict = {key: KCFTracker() for key in objects_list}
        elif track_method == "MOSSE":
            trackers_dict = {key: MOSSETracker() for key in objects_list}
        else:
            raise ValueError
        for item in objects_list:
            trackers_dict[item].init(frame, objects_detected[item][0])

    return stream, objects_detected, objects_list, trackers_dict

def postprocess(frame, out, threshold, classes, framework):
    """
    postprocess of CNN output
    :param frame:
    :param out: the prediction of cnn model
    :param threshold:
    :param classes:
    :param framework:
    :return: objects_detected: all objects detected, the bounding box coordinates of them, and the confidence of them
    """
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    objects_detected = dict()

    if framework == 'Caffe':
        for detection in out[0, 0]:
            confidence = detection[2]
            if confidence > threshold:
                left = int(detection[3] * frameWidth)
                top = int(detection[4] * frameHeight)
                right = int(detection[5] * frameWidth)
                bottom = int(detection[6] * frameHeight)
                # classId = int(detection[1]) - 1  # Skip background label

                classId = int(detection[1])
                i = 0
                label = classes[classId]
                label_with_num = str(label) + '_' + str(i)
                while (True):
                    if label_with_num not in objects_detected.keys():
                        break
                    label_with_num = str(label) + '_' + str(i)
                    i = i + 1
                objects_detected[label_with_num] = [(int(left), int(top), int(right - left), int(bottom - top)),
                                                    confidence]
                # print(label_with_num + ' at co-ordinates '+ str(objects_detected[label_with_num]))

    else:
        for detection in out:
            confidences = detection[5:]
            classId = np.argmax(confidences)
            confidence = confidences[classId]
            if confidence > threshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = center_x - (width / 2)
                top = center_y - (height / 2)

                i = 0
                label = classes[classId]
                label_with_num = str(label) + '_' + str(i)
                while (True):
                    if label_with_num not in objects_detected.keys():
                        break
                    label_with_num = str(label) + '_' + str(i)
                    i = i + 1
                objects_detected[label_with_num] = [(int(left), int(top), int(width), int(height)), confidence]
                # print(label_with_num + ' at co-ordinates '+ str(objects_detected[label_with_num]))

    return objects_detected

def drawPred(frame, objects_detected):
    """
    draw the bounding box on given image
    :param frame:
    :param objects_detected:
    :return:
    """

    for object_, info in objects_detected.items():
        box = info[0]
        confidence = info[1]
        label = '%s: %.2f' % (object_, confidence)
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv.rectangle(frame, p1, p2, (0, 255, 0))
        left = int(box[0])
        top = int(box[1])
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255),
                     cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))