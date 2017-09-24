import time
import R2D2
Driver = R2D2.Driver()
camara = R2D2.Vision()
stream = R2D2.Stream('192.168.1.100', 5000,streamer=1)
import cv2
collect_data = True



try:
    last_time = time.time()
    for frame in camara.camera.capture_continuous(camara.rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        stream.Send(gray_image)
        if collect_data:
            move = stream.Receive(5200)
            Driver.move2motor(move)
        else:
            print(round(1/(time.time() - last_time),2))
            last_time = time.time()
        camara.rawCapture.truncate(0)
finally:
    stream.stop()
