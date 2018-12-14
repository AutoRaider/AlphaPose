import fn
import json
import os
import cv2
import numpy as np
import torch

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

	result = []
	for i in range(len(data)):
		keypoints = data[i]['keypoints']
		dim = len(keypoints)
		kp_scores = []
		for i in range(len(keypoints)-1, 0, -3):
			kp_scores.append( keypoints.pop(i) )
		kp_scores.reverse()
		kp_scores = torch.FloatTensor(kp_scores)
		kp_preds = np.reshape(keypoints,(int(dim/3),2))
		kp_preds = torch.FloatTensor(kp_preds)
		human = {'keypoints':kp_preds, 'kp_score':kp_scores}
		result.append(human)
	all_result = {'imgname':data[0]['image_id'], 'result':result}
	return all_result


if __name__ == '__main__':

	origin = os.path.dirname(os.getcwd())
	target_json = "/alphapose/alphapose-results.json"
	path = origin + target_json
	
	test = read_json(path)
	image = cv2.imread('badminton.jpg')
	img = fn.vis_frame(image,test)

	cv2.imshow('img', img)
	cv2.waitKey(0)
