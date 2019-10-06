import time
import R2D2

Driver = R2D2.Driver()
camara = R2D2.Vision()
stream = R2D2.Stream(str(R2D2.config('IP')), int(R2D2.config('Port')), streamer=1)

# Do You Want to Collect Data or Test the Camara ?
collect_data = bool(R2D2.config('CollectData'))

try:
    last_time = time.time()
    while True:
        frame = camara.getFrame('COLOUR')
        stream.Send(frame)
        if collect_data:
            move = stream.Receive(5200)
            Driver.move2motor(move)
        else:
            print(round(1/(time.time() - last_time),2))
            last_time = time.time()

finally:
    stream.stop()
    Driver.clear()
