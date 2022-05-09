import cv2 as cv

img_flag = 1

if img_flag == 0:
    src_img = './facefile/face1.jpeg'
    des_img = './facefile/gray_face1.jpeg'
elif img_flag == 1:
    src_img = './facefile/big_guys.jpeg'
    des_img = './facefile/gray_big_guys.jpeg'


img = cv.imread(src_img)
# 灰度处理
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 保存图片
cv.imwrite(des_img, gray_img)
# 修改尺寸
# resize_grayimg = cv.resize(gray_img, dsize=(225, 400))
resize_grayimg = cv.resize(gray_img, dsize=(int(gray_img.shape[1]/gray_img.shape[0]*400), 400))

# 显示图片大小
print('BeforeSize:',gray_img.shape)
print('AfterSize:',resize_grayimg.shape)

cv.imshow('face', img)
cv.imshow('grayface', resize_grayimg)
cv.waitKey(0)
cv.destroyAllWindows()