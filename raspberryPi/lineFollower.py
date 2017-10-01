import R2D2

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
    stream = R2D2.Stream(str(R2D2.config('IP')), int(R2D2.config('Port')), streamer=1)
except:
    stream = None

LF = R2D2.LineFollower(collectData=True)
AI = R2D2.AI()


try:
    while True:
        frame = camara.getFrame('COLOR')
        stream.Send(frame)
        try:
            stream.Send(frame)
        except:
            pass
        _, _, sensorArray1, sensorArray2, sensorArray3, sensorArray4 = LF.getProcessedImage(frame)
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
        print(move, Driver.getDistance(), Driver.loaded, line1, line2, line3, line4)
        Driver.move2motor(move)

        LF.collectData(move, str(0), line1, line2, line3, line4, line5, line6, line7)

        if not Driver.loaded:
            if Driver.getDistance() < 9:
                Driver.setGPIO()
                while Driver.getDistance() > 7:
                    Driver.fwd(.005)
                Driver.load()
                Driver.loaded = 1

        camara.rawCapture.truncate(0)
finally:
    Driver.clear()