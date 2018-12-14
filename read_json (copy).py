import fn
import json
import os
import cv2
import numpy as np
import torch



origin = os.path.dirname(os.getcwd())
target_json = "/alphapose/alphapose-results.json"
origin = origin + target_json
print(origin)


# with open(origin, encoding='utf-8') as f:
#     line = f.readline()
#     d = json.loads(line)
#     print(len(d))
d = []
with open(origin, 'r', encoding='utf-8') as f:
	try:
	    while True:
	        line = f.readline()
	        if line:
	            d.extend(json.loads(line))
	            # print(len(d))
	        else:
	            break
	except:
	    f.close()
print('here:',len(d))

# image_id = d[0]['image_id']
# category_id = d[0]['category_id']
# score = d[0]['score']
keypoints = d[0]['keypoints']
print(type(d[0]))
print(type(keypoints))
print(len(keypoints))
# print(keypoints[15], keypoints[16])

image = cv2.imread('badminton.jpg')
# print(d[0])
# print( d[0]['image_id'] )
print('//////////////////////////////////')

result = []
for i in range(len(d)):
	keypoints = d[i]['keypoints']
	dim = len(keypoints)
	kp_scores = []
	for i in range(len(keypoints)-1, 0, -3):
		kp_scores.append( keypoints.pop(i) )
	kp_scores.reverse()
	kp_scores = torch.FloatTensor(kp_scores)

	kp_preds = np.reshape(keypoints,(int(dim/3),2))
	# kp_preds = list(kp_preds)
	kp_preds = torch.FloatTensor(kp_preds)
	human = {'keypoints':kp_preds, 'kp_score':kp_scores}
	result.append(human)
all_result = {'imgname':d[0]['image_id'], 'result':result}
# print(all_result['result'])
img = fn.vis_frame(image,all_result)

cv2.imshow('img', img)
cv2.waitKey(0)
print('//////////////////////////////////')

# kp_scores = []
# for i in range(len(keypoints)-1, 0, -3):
# 	kp_scores.append( keypoints.pop(i) )

# kp_scores.reverse()

# kp_preds = np.reshape(keypoints,(17,2))
# human = {'keypoints':kp_preds, 'kp_score':kp_scores}
print('///////////////////////')
# print(human['keypoints'])


# fn.vis_frame(image,d[0])

# print('\nimage_id:', image_id)
# print('\ncategory_id:', category_id)
# print('\nscore', score)
# print('\nkeypoints', keypoints)


