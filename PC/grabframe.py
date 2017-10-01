import time
import cv2
import numpy as np
import R2D2
from PC.getkeys import key_check


def getMove(keys):
    output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if 'W' in keys:
        output[0] = 1
    elif 'A' in keys:
        output[1] = 1
    elif 'D' in keys:
        output[2] = 1
    elif 'S' in keys:
        output[3] = 1
    elif 'I' in keys:
        output[4] = 1
    elif 'J' in keys:
        output[5] = 1
    elif 'L' in keys:
        output[6] = 1
    elif 'K' in keys:
        output[7] = 1
    elif 'Y' in keys:
        output[8] = 1
    elif 'G' in keys:
        output[9] = 1
    else:
        output[10] = 1
    return output

collectData = bool(R2D2.config('CollectData'))
stream = R2D2.Stream(str(R2D2.config('IP')), int(R2D2.config('Port')), streamer=0)
LF = R2D2.LineFollower(collectData=collectData)

if not collectData:
    AI = R2D2.AI()

last_time = time.time()
line1 = [];line2 = [];line3 = [];line4 = [];line5 = [];line6 = [];line7 = []
paused = True
loaded = 0

for _ in range(5):
    keys = key_check()
    move = getMove(keys)

while 1:
    try:
        frame = stream.Receive(1000000)
        img, display, sensorArray1, sensorArray2, sensorArray3, sensorArray4 = LF.getProcessedImage(frame)
        cv2.imshow('frame', frame)
        cv2.imshow('display', display)
        cv2.imshow('Processed Image', img)

        if collectData:
            keys = key_check()
            move = getMove(keys)
            stream.Send(move)
            if 'G' in keys:
                if loaded:
                    loaded = 0
                    time.sleep(1)
                else:
                    loaded = 1
                    time.sleep(1)

            if np.count_nonzero(move[:10]) != 0 and not paused:
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

                LF.collectData(np.argmax(move[:10]), str(loaded), line1, line2, line3, line4, line5, line6, line7)

            if 'P' in keys:
                if paused:
                    paused = False
                    print('unpausing')
                    time.sleep(1)
                else:
                    paused = True
                    print('Pausing')
                    time.sleep(1)
        else:
            print(round(1 / (time.time() - last_time), 2))
            last_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        if collectData:
            stream.Send([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])