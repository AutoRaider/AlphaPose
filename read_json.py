import fn
import json
import os
import cv2
import numpy as np
import torch
import copy

def read_json(path):
	data = []
	with open(path, 'r', encoding='utf-8') as f:
		try:
		    while True:
		        line = f.readline()
		        if line:
		            data.extend(json.loads(line))
		        else:
		            break
		except:
		    f.close()
	# the number of people in a sigle picture
	result = []
	# all of the picture capture from video
	all_result = []
	for i in range(len(data)):
		image_id = data[i]['image_id']
		keypoints = data[i]['keypoints']
		dim = len(keypoints)
		kp_scores = []
		for j in range(len(keypoints)-1, 0, -3):
			kp_scores.append( keypoints.pop(j) )
		kp_scores.reverse()
		kp_scores = torch.FloatTensor(kp_scores)
		kp_preds = np.reshape(keypoints,(int(dim/3),2))
		kp_preds = torch.FloatTensor(kp_preds)
		human = {'keypoints':kp_preds, 'kp_score':kp_scores}
		result.append(human)

		if i+1 == len(data) or image_id != data[i+1]['image_id']:
			single_result = {'imgname':image_id, 'result':result}
			all_result.append(copy.deepcopy(single_result))
			result.clear()

	return all_result


if __name__ == '__main__':

	origin = os.path.dirname(os.getcwd())
	target_json = "/alphapose/alphapose-results.json"
	path = origin + target_json

	test = read_json(path)
	print(len(test))
	print( (test[5]['imgname']) )
	# image = cv2.imread('badminton.jpg')
	# img = fn.vis_frame(image,test[0])

	# cv2.imshow('img', img)
	# cv2.waitKey(0)
