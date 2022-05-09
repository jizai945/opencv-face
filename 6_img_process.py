import os
import cv2
from PIL import Image
import numpy as np

# 人脸检测函数
def getImageAndLabels(path):
    facesSamples = []
    ids = []
    imagePaths=[os.path.join(path, f) for f in os.listdir(path)]
    # 导入分类器
    face_detector = cv2.CascadeClassifier('D:\\ProgramData\\Miniconda3\\envs\\opencv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml')
    # 遍历列表中的图片
    for imagePath in imagePaths:
        # 打开图片，黑白化
        PIL_img = Image.open(imagePath).convert('L')
        # 将图像转换为数组，以黑白深浅
        # PIL_img = cv2.resize(PIL_img, dsize=(400, 400))
        img_numpy = np.array(PIL_img, 'uint8')
        # 获取图片人脸特征
        faces = face_detector.detectMultiScale(img_numpy)
        # 获取每张图片的id和姓名
        id = int(os.path.split(imagePath)[1].split('.')[0])
        print(id)
        # 预防无面容照片
        for x, y, w, h in faces:
            ids.append(id)
            facesSamples.append(img_numpy[y:y+h, x:x+w])

        # 打印脸部特征和id
        print('id:', id)
        print('fs', facesSamples)
    return facesSamples, ids

if __name__ == '__main__':
    # 图片路径
    path = './saveface/'
    #获取图像数组和id标签数组和姓名
    faces,ids=getImageAndLabels(path)
    #获取训练对象
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces,np.array(ids))
    #保存文件
    recognizer.write('./trainer.yml')


