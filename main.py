from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import cv2 as cv
from PIL import Image, ImageTk
import numpy as np
import os, winreg, getpass, time, threading
from object_detection import object_detector, detect, postprocess, drawPred


class DirPath:
    def get_path(self):
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\{}\LPR".format(getpass.getuser()))
            self.path = winreg.QueryValueEx(key, "LPR")
        except:
            self.path = None
        return self.path

    def set_path(self, path):
        key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\{}\LPR".format(getpass.getuser()))
        winreg.SetValueEx(key, "LPR", 0, winreg.REG_SZ, path)
        self.path = path


class DetectTrackSurface(Tk):

    def __init__(self, model_name, confidence_thr, input=None):
        super().__init__()
        self.model_name = model_name
        self.confidence_thr = confidence_thr
        self.input = input
        self.title("LPR Surface")
        self.resizable(0, 0)
        self.label_pic_width = 500
        self.label_pic_height = 450
        self.button_width = 100
        self.button_height = 40
        self.text_width = 225
        self.text_height = 200
        self.tk_width = self.label_pic_width + self.button_width * 3
        self.tk_height = self.label_pic_height
        self.geometry(str(self.tk_width) + "x" + str(self.tk_height))
        self.track_method = "KCF"
        self.root = None
        self.video_run = False
        self.camera = False
        self.exit = False

        def label_init():
            # Picture Label:
            self.label_pic = Label(self)
            self.label_pic.place(x=0, y=0, width=self.label_pic_width, height=self.label_pic_height)

        def button_init():

            # Picture Button
            self.button_pic = Button(self, text="Load Image", command=self.load_picture)
            self.button_pic.place(x=self.label_pic_width + self.button_width // 2,
                                  y=2 * self.button_height / 2,
                                  width=self.button_width, height=self.button_height)
            # Video Button
            self.button_video = Button(self, text="Detect on Image", command=self.detect_image)
            self.button_video.place(x=self.label_pic_width + self.button_width // 0.6,
                                    y=2 * self.button_height / 2,
                                    width=self.button_width, height=self.button_height)

            # KCF Button
            self.button_kcf = Button(self, text="KCF Method ", command=self.KCF)
            self.button_kcf.place(x=self.label_pic_width + self.button_width // 2,
                                  y=5 * self.button_height / 2,
                                  width=self.button_width, height=self.button_height)

            # MOSSE Button
            self.button_mosse = Button(self, text="MOSSE Method ", command=self.MOSSE)
            self.button_mosse.place(x=self.label_pic_width + self.button_width // 0.6,
                                    y=5 * self.button_height / 2,
                                    width=self.button_width, height=self.button_height)

            # Camera Button
            self.button_camera = Button(self, text="Use Camera ", command=self.use_camera)
            self.button_camera.place(x=self.label_pic_width + self.button_width // 2,
                                     y=8 * self.button_height / 2,
                                     width=self.button_width, height=self.button_height)

            # Quit Button
            self.button_quit = Button(self, text="Quit", command=self.quit)
            self.button_quit.place(x=self.label_pic_width + self.button_width // 0.6,
                                   y=8 * self.button_height / 2,
                                   width=self.button_width, height=self.button_height)

        def text_init():
            self.text = Text(self)
            self.text.place(x=self.label_pic_width + self.button_width // 2,
                            y=11 * self.button_height / 2,
                            width=self.text_width, height=self.text_height)
        label_init()
        button_init()
        text_init()

        self.text.insert("end", " ---------init success---------")
        self.text.insert("end", '\n')
        self.mainloop()

    def load_model(self):
        if self.model_name == "yolo-v2":
            model = 'model_data/yolov2.weights'
            config = 'model_data/yolov2.cfg'
            classes = 'model_data/coco_classes.txt'
        elif self.model_name == "mobilenet-ssd":
            model = 'model_data/MobileNetSSD_deploy.caffemodel'
            config = 'model_data/MobileNetSSD_deploy.prototxt'
            classes = 'model_data/MobileNet_classes.txt'
        else:
            print("Please choose 'yolo-v2' or 'mobilenet-ssd' as your model name")
            raise ValueError
        predictor = object_detector(model, config)
        with open(classes, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')

        return predictor, classes

    def resize_picture(self, img_cv):
        if img_cv is None:
            messagebox.showerror(title="FILE ERROR", message="Please Input Correct Filename!")
            return None

        img_cvRGB = cv.cvtColor(img_cv, cv.COLOR_BGR2RGB)
        img = Image.fromarray(img_cvRGB)
        img_tk = ImageTk.PhotoImage(image=img)

        pic_width = img_tk.width()
        pic_height = img_tk.height()
        # print("Picture Size:", pic_width, pic_height)
        if pic_width <= self.label_pic_width and pic_height <= self.label_pic_height:
            return np.array(img), img_tk

        width_scale = 1.0 * self.label_pic_width / pic_width
        height_scale = 1.0 * self.label_pic_height / pic_height

        scale = min(width_scale, height_scale)

        resize_width = int(pic_width * scale)
        resize_height = int(pic_height * scale)

        img = img.resize((resize_width, resize_height), Image.ANTIALIAS)
        img_tk = ImageTk.PhotoImage(image=img)

        return np.array(img), img_tk

    # Load picture
    def load_picture(self):
        self.video_run = False

        dp = DirPath()
        if None == dp.get_path():
            init_path = ""
        else:
            init_path = dp.path

        # file_name = None
        file_name = filedialog.askopenfilename(title='Load Picture',
                                               filetypes=[('Picture File', '*.jfif *.jpg *.png *.gif'),
                                                          ('All Files', '*')],
                                               initialdir=init_path)

        self.text.insert("end", file_name)
        self.text.insert("end", '\n')
        if not os.path.isfile(file_name):
            messagebox.showerror(title="FILE ERROR", message="Please Input Correct Filename!")
            return False
        # Read Picture File.
        try:
            # self.img_ori = Image.open(file_name)
            # img_cv = cv.imdecode(np.fromfile(file_name, dtype=np.uint8), cv.IMREAD_COLOR)
            img_cv = cv.imread(file_name)
        except:
            messagebox.showerror(title="FILE ERROR", message="Image Open Failed!")
            return False

        dp.set_path(file_name)
        self.img, self.img_ori = self.resize_picture(img_cv)
        if self.img_ori is None:
            messagebox.showerror(title="FILE ERROR", message="Image Open Failed!")
            return False

        self.label_pic.configure(image=self.img_ori)

    def detect_image(self):
        if not self.img_ori:
            messagebox.showerror(title="ERROR", message="Image read Failed")
            return False
        else:
            predictor, classes = self.load_model()
            self.img = cv.cvtColor(self.img, cv.COLOR_BGR2RGB)
            predictions = predictor.predict(self.img)
            objects_detected = postprocess(self.img, predictions, self.confidence_thr, classes, predictor.framework)
            drawPred(self.img, objects_detected)
            _, self.img_ori = self.resize_picture(self.img)
            self.label_pic.configure(image=self.img_ori)

    # Video Thread
    def video_thread(self):
        self.video_run = True
        predictor, classes = self.load_model()
        if self.camera:
            try:
                stream = cv.VideoCapture(0)
                self.camera = False
            except:
                messagebox.showerror(title="CAMERA ERROR", message="Camera Open Failed!")
                self.video_run = False
                return False
        else:
            dp = DirPath()
            if dp.get_path() is None:
                init_path = ""
            else:
                init_path = dp.path

            # file_name = None
            file_name = filedialog.askopenfilename(title='Load video',
                                                   filetypes=[('Video type', '*.mp4 *.mkv *.flv *.rmvb'),
                                                              ('All Files', '*')],
                                                   initialdir=init_path)
            if not os.path.isfile(file_name):
                messagebox.showerror(title="FILE ERROR", message="Please Input Correct Filename!")
                self.video_run = False
                return False
            try:
                stream = cv.VideoCapture(file_name)
            except:
                messagebox.showerror(title="FILE ERROR", message="Video Open Failed!")
                self.video_run = False
                return False

            dp.set_path(file_name)
        stream, objects_detected, objects_list, trackers_dict = detect(
            stream, predictor, self.confidence_thr, classes, self.track_method)
        failed_track_flag = 1
        while stream.isOpened():
            grabbed, frame = stream.read()
            if not grabbed:
                messagebox.showerror(title="VIDEO ERROR", message="Video Read Failed!")
                self.video_run = False
                return False
            if len(objects_detected) > 0:
                del_items = []
                for obj, tracker in trackers_dict.items():
                    ok, bbox = tracker.update(frame)
                    if ok:
                        objects_detected[obj][0] = bbox
                    else:
                        self.text.insert("end", f"Failed to track {obj}")
                        self.text.insert("end", '\n')
                        del_items.append(obj)
                        failed_track_flag = 1

                for item in del_items:
                    trackers_dict.pop(item)
                    objects_detected.pop(item)

            if len(objects_detected) > 0:
                if failed_track_flag:
                    self.text.insert("end", f"Tracking {[item[0] for item in objects_detected.items()]}")
                    self.text.insert("end", '\n')
                    failed_track_flag = 0
                drawPred(frame, objects_detected)
            else:
                cv.putText(frame, 'Tracking Failure. Trying to detect more objects', (50, 80), cv.FONT_HERSHEY_SIMPLEX,
                           0.75, (0, 0, 255), 2)
                stream, objects_detected, objects_list, trackers_dict = detect(
                    stream, predictor, self.confidence_thr, classes, self.track_method)

            _, self.img_ori = self.resize_picture(frame)
            self.label_pic.configure(image=self.img_ori)
            self.label_pic.image = self.img_ori
            if self.exit:
                self.exit = False
                self.video_run = False
                break

        stream.release()
        self.label_pic.configure(image='')
        self.video_run = False
        self.text.insert("end", "Video Thread Finish!")
        self.text.insert("end", '\n')

    # Load Video From local disk
    def load_video(self):
        if self.video_run:
            messagebox.showerror(title="ERROR", message="Please Wait Until Last Video Ended")
            return None
        self.thread = threading.Thread(target=self.video_thread)
        self.thread.setDaemon(True)
        self.thread.start()
        self.video_run = True

    def use_camera(self):
        if self.video_run:
            messagebox.showerror(title="ERROR", message="Please Wait Until Last Video Ended")
            return None
        self.camera = True
        self.load_video()

    def KCF(self):
        self.track_method = "KCF"
        self.load_video()


    def MOSSE(self):
        self.track_method = "MOSSE"
        self.load_video()

    def quit(self):
        if self.video_run:
            self.exit = True
            return
        print("finish")
        self.destroy()
        return

if __name__ == '__main__':
    model_name = 'yolo-v2'
    confidence_thr = 0.35
    LS = DetectTrackSurface(model_name, confidence_thr)
