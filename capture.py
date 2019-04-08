#coding=utf-8
import cv2 
import time

def capture(model, fun):
    global img

    cv2.namedWindow('camera', 1)
    video = 'http://admin:admin@192.168.1.199:8081/'
    capture = cv2.VideoCapture(video)

    num = 0
    while True:
        success, img = capture.read()
        img = fun(model, img, 0)
        try:
            cv2.imshow('camera', img)
        except:
            print(img.shape)
            break
        key = cv2.waitKey(10)

        if key == 27:
            print("esc break...")
            break
        if key == ord(' '):
            num = num + 1
            filename = "frames_%s.jpg" % num
            cv2.imwrite(filename, img)
        if key == ord('a'):
            fun(model, img, 1)
            print('save\n')

    capture.release()
    cv2.destroyWindow('camera')