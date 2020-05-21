import numpy as np
import cv2
#import fhog


# fft tools
def fftd(img, backwards=False):
    return cv2.dft(np.float32(img), flags=(
        (cv2.DFT_INVERSE | cv2.DFT_SCALE) if backwards else cv2.DFT_COMPLEX_OUTPUT))  # 'flags =' is necessary!


def real(img):
    return img[:, :, 0]


def imag(img):
    return img[:, :, 1]


def complex_multiplication(a, b):
    res = np.zeros(a.shape, a.dtype)

    res[:, :, 0] = a[:, :, 0] * b[:, :, 0] - a[:, :, 1] * b[:, :, 1]
    res[:, :, 1] = a[:, :, 0] * b[:, :, 1] + a[:, :, 1] * b[:, :, 0]
    return res


def complex_division(a, b):
    res = np.zeros(a.shape, a.dtype)
    divisor = 1. / (b[:, :, 0] ** 2 + b[:, :, 1] ** 2)

    res[:, :, 0] = (a[:, :, 0] * b[:, :, 0] + a[:, :, 1] * b[:, :, 1]) * divisor
    res[:, :, 1] = (a[:, :, 1] * b[:, :, 0] + a[:, :, 0] * b[:, :, 1]) * divisor
    return res


def rearrange(img):
    # return np.fft.fftshift(img, axes=(0,1))
    assert (img.ndim == 2)
    img_ = np.zeros(img.shape, img.dtype)
    xh, yh = img.shape[1] // 2, img.shape[0] // 2
    img_[0:yh, 0:xh], img_[yh:2 * yh, xh:2 * xh] = img[yh:2 * yh, xh:2 * xh], img[0:yh, 0:xh]
    img_[0:yh, xh:2 * xh], img_[yh:2 * yh, 0:xh] = img[yh:2 * yh, 0:xh], img[0:yh, xh:2 * xh]
    return img_


# recttools
def x2(rect):
    return rect[0] + rect[2]


def y2(rect):
    return rect[1] + rect[3]


def limit(rect, limit):
    if (rect[0] + rect[2]) > (limit[0] + limit[2]):
        rect[2] = limit[0] + limit[2] - rect[0]
    if (rect[1] + rect[3]) > (limit[1] + limit[3]):
        rect[3] = limit[1] + limit[3] - rect[1]
    if rect[0] < limit[0]:
        rect[2] -= (limit[0] - rect[0])
        rect[0] = limit[0]
    if rect[1] < limit[1]:
        rect[3] -= (limit[1] - rect[1])
        rect[1] = limit[1]
    if rect[2] < 0:
        rect[2] = 0
    if rect[3] < 0:
        rect[3] = 0
    return rect


def get_border(original, limited):
    res = [0, 0, 0, 0]
    res[0] = limited[0] - original[0]
    res[1] = limited[1] - original[1]
    res[2] = x2(original) - x2(limited)
    res[3] = y2(original) - y2(limited)
    assert (np.all(np.array(res) >= 0))
    return res


def sub_window(img, window, border_type=cv2.BORDER_CONSTANT):
    cut_window = [x for x in window]
    limit(cut_window, [0, 0, img.shape[1], img.shape[0]])  # modify cut_window
    assert (cut_window[2] > 0 and cut_window[3] > 0)
    border = get_border(window, cut_window)
    res = img[cut_window[1]:cut_window[1] + cut_window[3], cut_window[0]:cut_window[0] + cut_window[2]]

    if border != [0, 0, 0, 0]:
        res = cv2.copyMakeBorder(res, border[1], border[3], border[0], border[2], border_type)
    return res


# KCF tracker
class KCFTracker:
    def __init__(self):
        self.lambdar = 0.0001  # regularization
        self.padding = 2.5  # extra area surrounding the target
        self.output_sigma_factor = 0.125  # bandwidth of gaussian target
        self.success = True

        self.interp_factor = 0.075
        self.sigma = 0.2
        self.cell_size = 1
        self.template_size = 1
        self.scale_step = 1

        self._tmpl_sz = [0, 0]  # cv::Size, [width,height]  #[int,int]
        self._roi = [0., 0., 0., 0.]  # cv::Rect2f, [x,y,width,height]  #[float,float,float,float]
        self.size_patch = [0, 0, 0]  # [int,int,int]
        self._scale = 1.  # float
        self._alphaf = None  # numpy.ndarray    (size_patch[0], size_patch[1], 2)
        self._prob = None  # numpy.ndarray    (size_patch[0], size_patch[1], 2)
        self._tmpl = None  # numpy.ndarray    raw: (size_patch[0], size_patch[1])
        self.hann = None  # numpy.ndarray    raw: (size_patch[0], size_patch[1])

    def sub_pixel_peak(self, left, center, right):
        divisor = 2 * center - right - left  # float
        return 0 if abs(divisor) < 1e-3 else 0.5 * (right - left) / divisor

    def create_hanning_mats(self):
        hann2t, hann1t = np.ogrid[0:self.size_patch[0], 0:self.size_patch[1]]

        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (self.size_patch[1] - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (self.size_patch[0] - 1)))
        hann2d = hann2t * hann1t

        self.hann = hann2d
        self.hann = self.hann.astype(np.float32)

    def create_gaussian_peak(self, sizey, sizex):
        syh, sxh = sizey / 2, sizex / 2
        output_sigma = np.sqrt(sizex * sizey) / self.padding * self.output_sigma_factor
        mult = -0.5 / (output_sigma * output_sigma)
        y, x = np.ogrid[0:sizey, 0:sizex]
        y, x = (y - syh) ** 2, (x - sxh) ** 2
        res = np.exp(mult * (y + x))
        return fftd(res)

    def gaussian_correlation(self, x1, x2):

        c = cv2.mulSpectrums(fftd(x1), fftd(x2), 0, conjB=True)  # 'conjB=' is necessary!
        c = fftd(c, True)
        c = real(c)
        c = rearrange(c)

        if (x1.ndim == 3) and (x2.ndim == 3):
            d = (np.sum(x1[:, :, 0] * x1[:, :, 0]) + np.sum(x2[:, :, 0] * x2[:, :, 0]) - 2.0 * c) / (
                    self.size_patch[0] * self.size_patch[1] * self.size_patch[2])
        elif (x1.ndim == 2) and (x2.ndim == 2):
            d = (np.sum(x1 * x1) + np.sum(x2 * x2) - 2.0 * c) / (
                    self.size_patch[0] * self.size_patch[1] * self.size_patch[2])

        d = d * (d >= 0)
        d = np.exp(-d / (self.sigma * self.sigma))

        return d

    def get_features(self, image, inithann, scale_adjust=1.0):
        extracted_roi = [0, 0, 0, 0]  # [int,int,int,int]
        cx = self._roi[0] + self._roi[2] / 2  # float
        cy = self._roi[1] + self._roi[3] / 2  # float

        if inithann:
            padded_w = self._roi[2] * self.padding
            padded_h = self._roi[3] * self.padding

            if self.template_size > 1:
                if padded_w >= padded_h:
                    self._scale = padded_w / float(self.template_size)
                else:
                    self._scale = padded_h / float(self.template_size)
                self._tmpl_sz[0] = int(padded_w / self._scale)
                self._tmpl_sz[1] = int(padded_h / self._scale)
            else:
                self._tmpl_sz[0] = int(padded_w)
                self._tmpl_sz[1] = int(padded_h)
                self._scale = 1.

            self._tmpl_sz[0] = int(self._tmpl_sz[0]) / 2 * 2
            self._tmpl_sz[1] = int(self._tmpl_sz[1]) / 2 * 2

        extracted_roi[2] = int(scale_adjust * self._scale * self._tmpl_sz[0])
        extracted_roi[3] = int(scale_adjust * self._scale * self._tmpl_sz[1])
        extracted_roi[0] = int(cx - extracted_roi[2] / 2)
        extracted_roi[1] = int(cy - extracted_roi[3] / 2)

        z = sub_window(image, extracted_roi, cv2.BORDER_REPLICATE)
        if (z.shape[1] != self._tmpl_sz[0]) or (z.shape[0] != self._tmpl_sz[1]):
            z = cv2.resize(z, tuple(map(int, self._tmpl_sz)))

        if z.ndim == 3 and z.shape[2] == 3:
            FeaturesMap = cv2.cvtColor(z,
                                       cv2.COLOR_BGR2GRAY)  # z:(size_patch[0], size_patch[1], 3)  FeaturesMap:(size_patch[0], size_patch[1])   #np.int8  #0~255
        elif (z.ndim == 2):
            FeaturesMap = z  # (size_patch[0], size_patch[1]) #np.int8  #0~255
        FeaturesMap = FeaturesMap.astype(np.float32) / 255.0 - 0.5
        self.size_patch = [z.shape[0], z.shape[1], 1]

        if (inithann):
            self.create_hanning_mats()  # create_hanning_mats need size_patch

        FeaturesMap = self.hann * FeaturesMap
        return FeaturesMap

    def detect(self, z, x):
        k = self.gaussian_correlation(x, z)
        res = real(fftd(complex_multiplication(self._alphaf, fftd(k)), True))

        _, pv, _, pi = cv2.minMaxLoc(res)  # pv:float  pi:tuple of int
        p = [float(pi[0]), float(pi[1])]  # cv::Point2f, [x,y]  #[float,float]

        if (pi[0] > 0) and (pi[0] < res.shape[1] - 1):
            p[0] += self.sub_pixel_peak(res[pi[1], pi[0] - 1], pv, res[pi[1], pi[0] + 1])
        if (pi[1] > 0) and (pi[1] < res.shape[0] - 1):
            p[1] += self.sub_pixel_peak(res[pi[1] - 1, pi[0]], pv, res[pi[1] + 1, pi[0]])

        p[0] -= res.shape[1] / 2.
        p[1] -= res.shape[0] / 2.

        return p, pv

    def train(self, x, train_interp_factor):
        k = self.gaussian_correlation(x, x)
        alphaf = complex_division(self._prob, fftd(k) + self.lambdar)

        self._tmpl = (1 - train_interp_factor) * self._tmpl + train_interp_factor * x
        self._alphaf = (1 - train_interp_factor) * self._alphaf + train_interp_factor * alphaf

    def init(self, image, roi):
        self._roi = list(map(float, roi))
        assert (roi[2] > 0 and roi[3] > 0)
        self._tmpl = self.get_features(image, 1)
        self._prob = self.create_gaussian_peak(self.size_patch[0], self.size_patch[1])
        self._alphaf = np.zeros((self.size_patch[0], self.size_patch[1], 2), np.float32)
        self.train(self._tmpl, 1.0)

    def update(self, image):
        if self._roi[0] + self._roi[2] <= 0:  self._roi[0] = -self._roi[2] + 1
        if self._roi[1] + self._roi[3] <= 0:  self._roi[1] = -self._roi[2] + 1
        if self._roi[0] >= image.shape[1] - 1:  self._roi[0] = image.shape[1] - 2
        if self._roi[1] >= image.shape[0] - 1:  self._roi[1] = image.shape[0] - 2

        cx = self._roi[0] + self._roi[2] / 2.
        cy = self._roi[1] + self._roi[3] / 2.

        loc, peak_value = self.detect(self._tmpl, self.get_features(image, 0, 1.0))
        if peak_value < 0.7:
            self.success = False

        if self.scale_step != 1:
            # Test at a smaller _scale
            new_loc1, new_peak_value1 = self.detect(self._tmpl, self.get_features(image, 0, 1.0 / self.scale_step))
            # Test at a bigger _scale
            new_loc2, new_peak_value2 = self.detect(self._tmpl, self.get_features(image, 0, self.scale_step))

            if (new_peak_value1 > peak_value) and (new_peak_value1 > new_peak_value2):
                loc = new_loc1
                peak_value = new_peak_value1
                self._scale /= self.scale_step
                self._roi[2] /= self.scale_step
                self._roi[3] /= self.scale_step
            elif new_peak_value2 > peak_value:
                loc = new_loc2
                peak_value = new_peak_value2
                self._scale *= self.scale_step
                self._roi[2] *= self.scale_step
                self._roi[3] *= self.scale_step
            else:
                loc = None
                self.success = False

                return self.success, self._roi

        self._roi[0] = cx - self._roi[2] / 2.0 + loc[0] * self.cell_size * self._scale
        self._roi[1] = cy - self._roi[3] / 2.0 + loc[1] * self.cell_size * self._scale

        if self._roi[0] >= image.shape[1] - 1:  self._roi[0] = image.shape[1] - 1
        if self._roi[1] >= image.shape[0] - 1:  self._roi[1] = image.shape[0] - 1
        if self._roi[0] + self._roi[2] <= 0:  self._roi[0] = -self._roi[2] + 2
        if self._roi[1] + self._roi[3] <= 0:  self._roi[1] = -self._roi[3] + 2
        assert (self._roi[2] > 0 and self._roi[3] > 0)

        x = self.get_features(image, 0, 1.0)
        self.train(x, self.interp_factor)

        return self.success, self._roi
