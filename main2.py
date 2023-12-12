import cv2

# 모델 가져오기, 해상도 좋게하는 모델
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel('models/EDSR_x3.pb')
# 해상도를 3배향상
sr.setModel('edsr', 3)

img = cv2.imread('imgs/06.jpg')

# 추론, 이미지 해상도 3배 향상 결과를 result에 저장해 주세요.
result = sr.upsample(img)
# 3배,3배씩 늘려 주세요
resized_img = cv2.resize(img, dsize=None, fx=3, fy=3)

cv2.imshow('img',img)
cv2.imshow('result',result)
cv2.imshow('resized_img',resized_img)
# 키입력할때까지 무한정 기다려라
cv2.waitKey(0)