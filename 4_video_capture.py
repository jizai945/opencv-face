import cv2 as cv
import time

def test(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_test = cv.CascadeClassifier('D:\\ProgramData\\Miniconda3\\envs\\opencv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml')
    face = face_test.detectMultiScale(gray_img)
    for x, y, w, h in face:
        cv.rectangle(img, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=2)
    cv.imshow('test', img)

# 调用摄像头
cap = cv.VideoCapture(0)

while True:
    flag,frame = cap.read()
    if not flag: #无值则退出
        break
    test(frame) #有值则调用函数
    if ord(' ') == cv.waitKey(1):
        break

cap.release()
cv.destroyAllWindows()