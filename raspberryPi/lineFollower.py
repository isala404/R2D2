import os

import R2D2
import cv2

line1 = []
line2 = []
line3 = []
line4 = []
line5 = []
line6 = []
line7 = []


Driver = R2D2.Driver()
camara = R2D2.Vision()
try:
    stream = R2D2.Stream('192.168.1.100', 5000, streamer=1)
except:
    stream = None


train = False

IM = R2D2.ImageProcessor()
AI = R2D2.AI(train=True)

if train:
    if not os.path.isfile('data/bot.csv'):
        f = open('data/bot.csv','w')
        count = 0
    else:
        f = open('data/bot.csv', 'r')
        count = sum(1 for _ in f) - 1
        f.close()
        f = open('data/bot.csv', 'a')

try:
    for frame in camara.camera.capture_continuous(camara.rawCapture, format="bgr", use_video_port=True):
        frame = frame.array
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            stream.Send(frame)
        except:
            pass
        _, _, _, sensorArray1, sensorArray2, sensorArray3, sensorArray4 = IM.getProcessedImage(frame)
        if not line1:
            line1 = sensorArray1
            line2 = sensorArray2
            line3 = sensorArray3
            line4 = sensorArray4
            line5 = sensorArray4
            line6 = sensorArray4
            line7 = sensorArray4
        else:
            line7 = line6
            line6 = line5
            line5 = line4
            line4 = sensorArray4
            line3 = sensorArray3
            line2 = sensorArray2
            line1 = sensorArray1

        data = [Driver.loaded] + line1 + line2 + line3 + line4 + line5 + line6 + line7
        move = AI.predict(data)
        print(move, Driver.getDistance(), Driver.loaded, line1, line2, line3, line4, line5)
        Driver.move2motor(move)

        if train:
            f.write(str(count)+','+str(move))
            for i in data:
                f.write(','+str(i))
            f.write('\n')
            count +=1

        if not Driver.loaded:
            if Driver.getDistance() < 9:
                Driver.setGPIO()
                while Driver.getDistance() > 7:
                    Driver.fwd(.005)
                Driver.load()
                Driver.loaded = 1

        camara.rawCapture.truncate(0)
finally:
    f.close()
    Driver.clear()