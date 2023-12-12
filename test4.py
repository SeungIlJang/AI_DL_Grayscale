#openCv 가져오기
import cv2
import numpy as np

# 컬러 복원 + 해상도 향상

# 그레이스케일 -> 칼라
# 모델 불러오기
proto = 'models/colorization_deploy_v2.prototxt'
weights = 'models/colorization_release_v2.caffemodel'
# 딥러닝 모델을 만들때 쓰는 프레임워크
net = cv2.dnn.readNetFromCaffe(proto, weights)
pts_in_hull = np.load('models/pts_in_hull.npy')
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1).astype(np.float32)
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full((1, 313), 2.606, np.float32)]

# ----- 해상도 시작
# 모델 가져오기, 해상도 좋게하는 모델
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel('models/test/EDSR_x4.pb')
# 해상도를 4배 향상
sr.setModel('edsr', 4)

img = cv2.imread('imgs/test/07.jpg')
h, w, c = img.shape
print(h, w, c)
# 1/3배,1/3배씩 줄여 주세요
resized_img = cv2.resize(img, dsize=None, fx=1/4, fy=1/4)
h, w, c = resized_img.shape
print(h, w, c)
# 추론, 이미지 해상도 4배 결과를 result에 저장
# 사이즈 줄였는데 원복사이즈로 출렴됨? 왜지?
result = sr.upsample(resized_img)
# ----- 해상도 끝

# 이미지 형태
h, w, c = result.shape
print(h, w, c)
# ------- # 그레이스케일 -> 칼라 Lab 시작
# 이미지 카피
img_input = result.copy()
#이미지 전처리,이미지는 uint8,float32만 사용
img_input = img_input.astype('float32') / 255.
# lab 형태로 변경
img_lab = cv2.cvtColor(img_input, cv2.COLOR_BGR2Lab)
# L채널만 추출
img_l = img_lab[:, :, 0:1]
#전치리 처리
blob = cv2.dnn.blobFromImage(img_l, size=(224, 224), mean=[50, 50, 50])

# input
net.setInput(blob)
# output
output = net.forward()

# 후처리 ,우리가 알아볼수 있는 형태
output = output.squeeze().transpose((1,2,0))
# 원본이미지 형태로 다시 변경,전처리시 리사이즈로 변형한걸 다시 원복
output_resized = cv2.resize(output,(w,h))

#L과 합쳐주기 
output_lab = np.concatenate([img_l,output_resized], axis=2)
#BGR로 변환
output_bgr = cv2.cvtColor(output_lab,cv2.COLOR_Lab2BGR)
#후처리 255곱해주기
output_bgr = output_bgr * 255
#보기 싫은것 잘라내기
output_bgr = np.clip(output_bgr,0,255)
# 정수형태로 변환
output_bgr = output_bgr.astype('uint8')
# ------- # 그레이스케일 -> 칼라 Lab 끝

cv2.imshow('img',img)
# cv2.imshow('resized_img',resized_img)
cv2.imshow('result',result)
cv2.imshow('output_bgr',output_bgr)
# 키입력할때까지 무한정 기다려라
cv2.waitKey(0)