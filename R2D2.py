import configparser
import time
import cv2
import socket
from _pickle import dumps, loads
import numpy as np
import os
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def config(sub, section='R2D2'):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config[section][sub]


class Driver(object):
    def __init__(self):
        self.gpio = __import__('RPi.GPIO')
        self.gpio = self.gpio.GPIO

        # Motor controller l298n
        self.enable_L = 36
        self.motorL_1 = 33
        self.motorL_2 = 35
        self.enable_R = 37
        self.motorR_1 = 38
        self.motorR_2 = 40

        # 2 LED
        self.led = 29
        self.led2 = 31

        # Servo motor (Arm)
        self.ser1 = 7
        self.ser2 = 11

        # hc-sr04 ultrasonic distance sensor
        self.TRIG = 12
        self.ECHO = 13

        self.gpio.setwarnings(TRUE)
        self.gpio.setmode(self.gpio.BOARD)

        self.gpio.setup(self.enable_L, self.gpio.OUT)
        self.gpio.setup(self.motorL_1, self.gpio.OUT)
        self.gpio.setup(self.motorL_2, self.gpio.OUT)
        self.gpio.setup(self.enable_R, self.gpio.OUT)
        self.gpio.setup(self.motorR_1, self.gpio.OUT)
        self.gpio.setup(self.motorR_2, self.gpio.OUT)

        self.gpio.setup(self.led, self.gpio.OUT)
        self.gpio.setup(self.led2, self.gpio.OUT)

        self.gpio.setup(self.ser1, self.gpio.OUT)
        self.gpio.setup(self.ser2, self.gpio.OUT)

        self.gpio.output(self.enable_L, False)
        self.gpio.output(self.motorL_1, False)
        self.gpio.output(self.motorL_2, False)
        self.gpio.output(self.enable_R, False)
        self.gpio.output(self.motorR_1, False)
        self.gpio.output(self.motorR_2, False)

        self.gpio.output(self.led, True)
        self.gpio.output(self.led2, True)

        self.gpio.setup(self.TRIG, self.gpio.OUT)
        self.gpio.setup(self.ECHO, self.gpio.IN)
        self.gpio.output(self.TRIG, False)
        self.loaded = 0
        self.unload()

    def setGPIO(self, EL=False, ML1=False, ML2=False, ER=False, MR1=False, MR2=False):
        self.gpio.output(self.enable_L, EL)
        self.gpio.output(self.motorL_1, ML1)
        self.gpio.output(self.motorL_2, ML2)
        self.gpio.output(self.enable_R, ER)
        self.gpio.output(self.motorR_1, MR1)
        self.gpio.output(self.motorR_2, MR2)

    def fwd(self, tf=.012):
        self.setGPIO(ER=True, MR1=True, EL=True, ML1=True)
        time.sleep(tf)
        self.setGPIO()

    def bwd(self, tf=.012):
        self.setGPIO(ER=True, MR2=True, EL=True, ML2=True)
        time.sleep(tf)
        self.setGPIO()

    def left(self, tf=.012):
        self.setGPIO(ER=True, MR1=True)
        time.sleep(tf)
        self.setGPIO()

    def right(self, tf=.012):
        self.setGPIO(EL=True, ML1=True)
        time.sleep(tf)
        self.setGPIO()

    def left90(self, tf=.01):
        self.setGPIO(ER=True, MR1=True, EL=True, ML2=True)
        time.sleep(tf)
        self.setGPIO()

    def right90(self, tf=.01):
        self.setGPIO(ER=True, MR2=True, EL=True, ML1=True)
        time.sleep(tf)
        self.setGPIO()

    def end(self, tf=.7):
        self.setGPIO()
        time.sleep(tf)

    def move2motor(self, moves):
        if type(moves) == list:
            move = moves.index(1)
        else:
            move = int(moves)
        if move == 0:
            self.fwd()
        elif move == 1:
            self.left()
        elif move == 2:
            self.right()
        elif move == 3:
            self.bwd()
        elif move == 4:
            self.fwd(0.15)
        elif move == 5:
            self.left90()
        elif move == 6:
            self.right90()
        elif move == 7:
            self.left90(0.3)
        elif move == 8:
            self.end()
        elif move == 9:
            if self.loaded:
                self.unload()
                time.sleep(2)
            else:
                self.load()
                time.sleep(2)

    def servoStart(self):
        self.servo1 = self.gpio.PWM(self.ser1, 50)
        self.servo2 = self.gpio.PWM(self.ser2, 50)
        self.servo1.start(12.0)
        self.servo2.start(1.9)

    def servoStop(self):
        self.servo1.stop()
        self.servo2.stop()

    def servoLoad(self):
        self.servo1.ChangeDutyCycle(2.25)
        self.servo2.ChangeDutyCycle(10)

    def servoUnLoad(self):
        self.servo1.ChangeDutyCycle(12.0)
        self.servo2.ChangeDutyCycle(1.9)

    def load(self):
        self.loaded = True
        self.servoStart()
        time.sleep(1)
        self.servoLoad()
        time.sleep(1)
        self.servoStop()

    def unload(self):
        self.loaded = False
        self.servoStart()
        time.sleep(1)
        self.servoUnLoad()
        time.sleep(1)
        self.servoStop()

    def getDistance(self):
        self.gpio.output(self.TRIG, True)
        time.sleep(0.00001)
        self.gpio.output(self.TRIG, False)

        while self.gpio.input(self.ECHO) == 0:
            self.pulse_start = time.time()

        while self.gpio.input(self.ECHO) == 1:
            self.pulse_end = time.time()

        self.pulse_duration = self.pulse_end - self.pulse_start

        self.distance = self.pulse_duration * 17150

        self.distance = round(self.distance, 2)

        return self.distance

    def clear(self):
        self.gpio.cleanup()


class Stream(object):
    def __init__(self, ip, port, streamer, rc=False):
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if streamer:
            self.conn.connect((ip, port))
        else:
            try:
                self.conn.bind((ip, port))
            except socket.error:
                print('Bind failed')
            self.conn.listen(5)
            print('Socket awaiting handshake')
            (self.conn, _) = self.conn.accept()
            print('Connected')
        self.rc = rc

    def Send(self, frame):
        self.conn.send(dumps(frame))

    def Receive(self, bitRate=1000000):
        data = self.conn.recv(bitRate)
        return loads(data)

    def stop(self):
        self.conn.close()


class Vision(object):
    def __init__(self):
        self.PiRGBArray = __import__('picamera.array').array.PiRGBArray
        self.PiCamera = __import__('picamera').PiCamera
        self.WIDTH = int(config('CapWidth'))
        self.HEIGHT = int(config('CapHeight'))
        self.camera = self.PiCamera()
        self.camera.resolution = (self.WIDTH, self.HEIGHT)
        self.camera.framerate = 70
        self.rawCapture = self.PiRGBArray(self.camera, size=(self.WIDTH, self.HEIGHT))
        time.sleep(0.2)

    def getFrame(self, color=str(config('CapWidth'))):
        for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
            image = frame.array
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.rawCapture.truncate(0)
            if color == 'Gray':
                return gray_image
            else:
                return image

    def seeFrame(self):
        for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
            image = frame.array
            self.rawCapture.truncate(0)
            cv2.imshow("Frame", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break


class LineFollower(object):
    def __init__(self, collectData=False):
        self.WIDTH = int(config('FrameWidth', 'LineFollower'))
        self.HEIGHT = int(config('FrameHeight', 'LineFollower'))
        self.start_x = int(config('StartX', 'LineFollower'))
        self.start_y = int(config('StartY', 'LineFollower'))
        self.end_x = int(config('EndX', 'LineFollower'))
        self.end_y = int(config('EndY', 'LineFollower'))
        self.gapBetweenLine = int(config('GapBetweenLine', 'LineFollower'))
        self.box_height = int(config('BoxHeight', 'LineFollower'))
        self.blackThreashHold = int(config('blackThreashHold', 'LineFollower'))
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        if collectData:
            if not os.path.isfile('data/data.csv'):
                self.f = open('data/data.csv', 'w')
                self.f.write('id,move,box_loaded')
                for i in range(1, 8):
                    for k in range(1, 7):
                        self.f.write(',line{}_{}'.format(i, k))
                self.f.write('\n')
                self.count = 0
                self.f.close()
                self.f = open('data/data.csv', 'a')
            else:
                self.f = open('data/data.csv', 'r')
                self.count = sum(1 for _ in self.f) - 1
                self.f.close()
                self.f = open('data/data.csv', 'a')

    def sensorArray(self, ROI, hm=int(config('Boxes', 'LineFollower'))):
        line = []
        for i in range(hm):
            size = (self.end_x - self.start_x) / hm
            start = int(i * size)
            end = int(i * size + size)
            box = ROI[:, start:end]
            area = box.shape[0] * box.shape[1]
            nonzero = np.count_nonzero(box)
            if int(area - nonzero) > int(area * 0.3):
                line.append(1)
            else:
                line.append(0)
        return line

    def getY(self, i, gap):
        return (self.end_y - self.start_y) - (self.box_height * i) + (self.gapBetweenLine * gap)

    def drawBoxes(self, img, lines, hm=int(config('Boxes', 'LineFollower'))):
        for idx, line in enumerate(lines):
            for i in range(hm):
                size = (self.end_x - self.start_x) / hm
                start = int(i * size)
                end = int(i * size + size)
                if line[i]:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                if idx == 0:
                    cv2.rectangle(img, (start, self.getY(7, 3)), (end, self.getY(6, 3)), color, 1)
                elif idx == 1:
                    cv2.rectangle(img, (start, self.getY(5, 2)), (end, self.getY(4, 2)), color, 1)
                elif idx == 2:
                    cv2.rectangle(img, (start, self.getY(3, 1)), (end, self.getY(2, 1)), color, 1)
                else:
                    cv2.rectangle(img, (start, self.getY(1, 0)), (end, self.getY(0, 0)), color, 1)
        return img

    def getProcessedImage(self, frame):

        frame = frame[self.start_y:self.end_y, self.start_x:self.end_x]
        display = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(frame, self.blackThreashHold, 255, cv2.THRESH_BINARY)

        ROISensorArray1 = img[self.getY(7, 3):self.getY(6, 3), :]
        ROISensorArray2 = img[self.getY(5, 2):self.getY(4, 2), :]
        ROISensorArray3 = img[self.getY(3, 1):self.getY(2, 1), :]
        ROISensorArray4 = img[self.getY(1, 0):self.getY(0, 0), :]

        sensorArray1 = self.sensorArray(ROISensorArray1)
        sensorArray2 = self.sensorArray(ROISensorArray2)
        sensorArray3 = self.sensorArray(ROISensorArray3)
        sensorArray4 = self.sensorArray(ROISensorArray4)

        display = cv2.resize(self.drawBoxes(display, [sensorArray1, sensorArray2, sensorArray3, sensorArray4]),
                             (240, 180))

        return img, display, sensorArray1, sensorArray2, sensorArray3, sensorArray4

    def collectData(self, move, *lines):
        print(self.count)
        try:
            self.f.write('{0},{1}'.format(str(self.count), move))
            for line in lines:
                for idx in line:
                    self.f.write(',{0}'.format(str(idx)))
            self.f.write('\n')
            self.count += 1
        except Exception as e:
            print(e)

        if self.count % 100 == 0:
            print('Saving Collected Data')
            self.f.close()
            self.f = open('data/data.csv', 'a')


class AI(object):
    def __init__(self):
        if os.path.isfile('data/model.pkl'):
            print('Model Loaded')
            self.clf = joblib.load('data/model.pkl')
        else:
            self.trainModel()

    def trainModel(self):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        print('Training Started')

        df = pd.read_csv('data/data.csv')

        df.drop(['id'], 1, inplace=True)

        X = np.array(df.drop([config('LableName')], axis=1))
        y = np.array(df[config('LableName')])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

        clf = LogisticRegression()

        clf.fit(X_train, y_train)

        accuracy = float(clf.score(X_test, y_test))
        print('Model accuracy is', accuracy * 100, '%')
        joblib.dump(clf, 'data/model.pkl')
        self.clf = clf

    def predict(self, data):
        data = np.array(data)
        prediction = self.clf.predict(data.reshape(1, -1))
        return prediction[0]
