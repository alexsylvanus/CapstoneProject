import create_wave
import threading
from time import sleep

def detection_method_1(lock):
    while True:
        # call detection_method_1.main() here
        lock.acquire()
        create_wave.main(50, 50, 'output1')
        lock.release()
        sleep(1)


def detection_method_2(lock):
    while True:
        # call detection_method_2.main() here
        lock.acquire()
        create_wave.main(50, 50, 'output2')
        lock.release()
        sleep(1)

if __name__ == '__main__':
    # create threads here
    lock = threading.Lock()

    t1 = threading.Thread(target=detection_method_1, args=(lock,))
    t2 = threading.Thread(target=detection_method_2, args=(lock,))

    t1.start()
    t2.start()

    t1.join()
    t2.join()
            