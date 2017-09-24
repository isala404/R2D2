import time
import cv2
import numpy as np
import R2D2
from PC.getkeys import key_check


def keys_to_output(keys):
    '''
    Source - https://github.com/Sentdex/pygta5
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    '''
    if 'W' in keys:
        output = w
    elif 'A' in keys:
        output = a
    elif 'D' in keys:
        output = d
    elif 'I' in keys:
        output = i
    elif 'J' in keys:
        output = j
    elif 'L' in keys:
        output = l
    elif 'K' in keys:
        output = k
    elif 'Y' in keys:
        output = y
    elif 'S' in keys:
        output = s
    elif 'G' in keys:
        output = g
    else:
        output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    return output


w = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
a = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
d = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
i = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
j = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
l = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
k = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
y = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
s = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
g = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

loaded = 0

stream = R2D2.Stream('192.168.1.100', 5000, streamer=0)
IM = R2D2.ImageProcessor()
AI = R2D2.AI()

last_time = time.time()

line1 = []
line2 = []
line3 = []
line4 = []
line5 = []
line6 = []
line7 = []

paused = True
collectData = False

for _ in range(5):
    keys = key_check()
    move = keys_to_output(keys)

while 1:
    try:
        frame = stream.Receive(1000000)
        frame, img, display, sensorArray1, sensorArray2, sensorArray3, sensorArray4 = IM.getProcessedImage(frame)
        cv2.imshow('frame', frame)
        cv2.imshow('display', display)
        cv2.imshow('Processed Image', img)

        if collectData:
            keys = key_check()
            move = keys_to_output(keys)
            stream.Send(move)
            if 'G' in keys:
                if loaded:
                    loaded = 0
                    time.sleep(1)
                else:
                    loaded = 1
                    time.sleep(1)

            if np.count_nonzero(move[:8]) != 0 and not paused:
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

                AI.collectData(np.argmax(move[:8]), str(loaded), line1, line2, line3, line4, line5, line6, line7)

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