import sys
import cv2

img = cv2.imread('cat.bmp', cv2.IMREAD_GRAYSCALE)

# 에러 확인
if img is None:
    print('Image load failed!')
    sys.exit()



# 창을 띄움
# cv2.namedWindow('image')
cv2.imshow('image', img)
cv2.waitKey()
# cv2.destroyAllWindows()  # 창을 닫기