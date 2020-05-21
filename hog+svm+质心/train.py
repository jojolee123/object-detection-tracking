import numpy as np
from sklearn import svm
import sklearn
import joblib
import cv2
from sklearn.svm import LinearSVC


def prepare_data(filepath):
    data = np.loadtxt(filepath, dtype=np.float32)
    np.random.shuffle(data)
    train = data[:, :-1]
    label = data[:, -1:]
    # train, label = sklearn.utils.shuffle(train, label)
    return train, label


def k_fold(data, label , k=0, random_state=42):
    train_data, valid_data, train_label, valid_label = sklearn.model_selection.train_test_split(
        data, label, test_size=0.1, random_state=random_state)
    return train_data, train_label, valid_data, valid_label


def train_one_fold(train_data, train_label, fold=0):
    classifier = LinearSVC(C=0.2, )
    classifier.fit(train_data, train_label.ravel())
    # svm = cv2.ml.SVM_create()
    # svm.setCoef0(0)
    # svm.setCoef0(0.0)
    # svm.setDegree(3)
    # criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
    # svm.setTermCriteria(criteria)
    # svm.setGamma(0)
    # svm.setKernel(cv2.ml.SVM_LINEAR)
    # svm.setNu(0.5)
    # svm.setP(0.1)  # for EPSILON_SVR, epsilon in loss function?
    # svm.setC(0.01)  # From paper, soft classifier
    # svm.setType(cv2.ml.SVM_EPS_SVR)  # C_SVC # EPSILON_SVR # may be also NU_SVR # do regression task
    # svm.train(train_data, cv2.ml.ROW_SAMPLE, train_label)
    return classifier


def valid(classifier, train_data, train_label, valid_data, valid_label):
    train_pred = classifier.predict(train_data)
    valid_pred = classifier.predict(valid_data)
    # train_pred[train_pred > 0.6] = 1
    # train_pred[train_pred <= 0.6] = 0
    # valid_pred[valid_pred > 0.6] = 1
    # valid_pred[valid_pred <= 0.6] = 0
    train_score = sklearn.metrics.accuracy_score(train_label, train_pred)
    valid_score = sklearn.metrics.accuracy_score(valid_label, valid_pred)
    # train_score = sklearn.metrics.auc(train_pred, train_label)
    # valid_score = sklearn.metrics.auc(valid_pred, valid_label)
    print(f'train score: {train_score}, valid score: {valid_score}')


def save_model(classifier, name='svm_fold', fold=0):
    joblib.dump(classifier, filename=f'model/{name}{fold}.pkl')


def load_model(fold=0):
    cls = joblib.load(filename=f'model/svm_fold{fold}.pkl')
    return cls


if __name__ == "__main__":
    filepath = 'train.txt'
    train, label = prepare_data(filepath)
    train_data, train_label, valid_data, valid_label = k_fold(train, label)
    clf = train_one_fold(train_data, train_label)
    valid(clf, train_data, train_label, valid_data, valid_label)
    save_model(clf, name='cv')