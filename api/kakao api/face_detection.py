import requests

url = "https://kapi.kakao.com/v1/vision/face/detect"
MYAPP_KEY = '**************************************'
headers = {'Authorization': 'KakaoAK {}'.format(MYAPP_KEY)}

filename = './face_detection_test_MR.jpg'
files = { 'file' : open(filename, 'rb')}

response = requests.post(url, headers=headers, files=files)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
%matplotlib inline

result = response.json()
faces = result['result']['faces'][0]
facial_points = faces['facial_points']
fig_w, fig_h = result['result']['width'], result['result']['height']

img = mpimg.imread('face_detection_test_MR.jpg')
fig,ax = plt.subplots(figsize=(10,10))

target_obj = ['right_eyebrow', 'left_eyebrow', 'jaw', 'right_eye', 'left_eye']
for each_obj in target_obj:
    for each in facial_points[each_obj]:
        rect_face = patches.Circle((each[0]*fig_w, each[1]*fig_h),linewidth=3, edgecolor='c')
        ax.add_patch(rect_face)
    
ax.imshow(img)
plt.show()