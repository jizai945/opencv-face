import cv2 as cv

gray_img = cv.imread('./facefile/gray_face1.jpeg')

resize_grayimg = cv.resize(gray_img, dsize=(225, 400))

x, y, w, h = (70, 110, 110, 120)
# 矩形绘制
cv.rectangle(resize_grayimg, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=2)
# 圆形绘制
cv.circle(resize_grayimg, center=(x+w//2, y+h//2), radius=60, color=(255, 0, 0), thickness=2)

cv.imshow('resize', resize_grayimg)
cv.waitKey(0)
cv.destroyAllWindows()