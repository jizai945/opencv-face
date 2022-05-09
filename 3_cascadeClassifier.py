# 级联分类器

import cv2 as cv

img_flag = 1
if img_flag == 0:
    src_img = './facefile/gray_face1.jpeg'
elif img_flag == 1:
    src_img = './facefile/gray_big_guys.jpeg'

gray_img = cv.imread(src_img)
resize_grayimg = cv.resize(gray_img, dsize=(int(gray_img.shape[1]/gray_img.shape[0]*600), 600))

# 定义测试函数
def test():
    # 通过命令可以查找cv2的路径 pip show opencv-python
    # 定义级联分类器的路径
    face_test = cv.CascadeClassifier('D:\\ProgramData\\Miniconda3\\envs\\opencv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml')
    # 对图像进行检测, 1.05为检测的倍数, 5为检测次数, 0为默认参数, 后面两个为最小检测范围和最大检测范围的像素
    face = face_test.detectMultiScale(resize_grayimg, 1.05, 10, 0, (20,20), (200, 200))
    for x, y, w, h in face:
        cv.rectangle(resize_grayimg, (x, y), (x+w, y+h), color = (0, 0, 255), thickness=2)
    cv.imshow('LYF', resize_grayimg)


test()
cv.waitKey(0)
cv.destroyAllWindows()