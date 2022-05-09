import cv2 as cv
import time
import os

# 调用摄像头
cap = cv.VideoCapture(0)

if not os.path.exists('./saveface'):
    os.mkdir('./saveface')

num = 1
while (cap.isOpened()):
    cap_flag, Vshow = cap.read()
    cv.imshow('testface', Vshow)
    a = cv.waitKey(1)
    if a == ord('s'): #按下s保存
        if cv.imwrite('./saveface/' +str(num)+ '.jpg',Vshow):
            print('第'+str(num)+'张保存完毕！！')
            num = num +1
        else:
            print('第'+str(num)+'张保存失败！！')
    elif a == ord(' ') or a == ord('q'):
        break

cap.release()
cv.destroyAllWindows()