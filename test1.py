import cv2
import numpy as np

# 그레이스케일 이미지 칼라로 변환

# 모델 불러오기
proto = 'models/colorization_deploy_v2.prototxt'
weights = 'models/colorization_release_v2.caffemodel'

# 딥러닝 모델을 만들때 쓰는 프레임워크
net = cv2.dnn.readNetFromCaffe(proto, weights)

pts_in_hull = np.load('models/pts_in_hull.npy')
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1).astype(np.float32)
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull]

net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full((1, 313), 2.606, np.float32)]

# 이미지 저장
img = cv2.imread('imgs/test/pexels-kate.jpg')
# 이미지 형태
h, w, c = img.shape
# 이미지가 너무커서 리사이즈함
img = cv2.resize(img,(int(h/5),int(w/5)))
h, w, c = img.shape
# 이미지 카피
img_input = img.copy()
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
# 원본이미지 형태로 다시 변경,전처리시 리사이즈로 변형한걸 다시 원복\
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

# 이미지 띄우기
cv2.imshow('img',img) 
cv2.imshow('output',output_bgr)
# 윈도우가 켜진 상태로 유지 # 키입력할때까지 무한정 기다려라
cv2.waitKey(0)



