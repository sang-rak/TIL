import sys
import cv2


src = cv2.imread('namecard1.jpg')

if src is None:
    print('image load failed')
    sys.exit()

# src = cv2.resize(src, (640, 480)) # 사진 사이즈 픽셀 조정
src = cv2.resize(src, (0, 0), fx=0.5, fy=0.5) # 사진 사이즈 비율 조정

src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

th, src_bin = cv2.threshold(src_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # OTSU알고리즘으로 자동결정 0, 255의미없음

print(th)

contours, _ = cv2.findContours(src_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

print(len(contours))


# 넓은 픽셀값은 살리고 폴리곤 라인에 빨간색 라인
for pts in contours:
    if cv2.contourArea(pts) < 1000:
        continue

    approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True)*0.02, True)  # 

    if len(approx) != 4:
        continue

    cv2.polylines(src, pts, True, (0, 0, 255))


cv2.imshow('src', src)
cv2.imshow('src_gray', src_gray)
cv2.imshow('src_bin', src_bin)
cv2.waitKey()
cv2.destroyAllWindows()

# 영상의 기하학적 변환 시작