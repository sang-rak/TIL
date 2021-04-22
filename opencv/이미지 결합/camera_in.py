import sys
import cv2


cap = cv2.VideoCapture(0) # 웹캠
cap = cv2.VideoCapture('vtest.avi') # 동영상

if not cap.isOpened():
    print('camera open failed')
    sys.exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    edge = cv2.Canny(frame, 50, 150) # 엣지 동영상 처리

    cv2.imshow('frame', frame)
    cv2.imshow('edge', edge)
    if cv2.waitKey(30) == 27: # 정지 ESC 
        break

cap.release()
cv2.destroyAllWindows()