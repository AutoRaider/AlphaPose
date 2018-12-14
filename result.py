import cv2
import os
import numpy
import sys
import read_json
import fn
origin = os.path.dirname(os.getcwd())
sys.path.insert(0,origin + '/tennis')
import geometry as ge
from calibration import Calibrater

def find_athlete(origin_data, transf_matrix, garget_width, target_height):
    # you can get the target dataset: origin_data
    img_final_list = []
    for i in range( len(origin_data['result'])-1, -1, -1 ):
        mean_point = (origin_data['result'][i]['keypoints'][15] + origin_data['result'][i]['keypoints'][16]) / 2
        people_position = ge.Point(int(mean_point[0]), int(mean_point[1]))
        people_transf_position = people_position.perspective(transf_matrix)
        x = people_transf_position.x
        y = people_transf_position.y
        if x<0 or x>garget_width or y<0 or y>target_height :
            del origin_data['result'][i]
        else:
            img_final_list.append(people_transf_position)
    return img_final_list


if __name__ == '__main__':
    # people_position = ge.Point(394,472)
    target_path = "/alphapose/badminton.jpg"
    target_img = origin + target_path
    image = cv2.imread(target_img)
    cv2.imshow('image',image)
    cv2.waitKey()
    size = image.shape # height and width
    print('origin img size:', size)
    tennis_width = 300
    tennis_height = 400

    # flag = int(input("If you have the transform matrix? \nYes: press '1', No:press '2'\n"))
    # if flag == 2:
    #     cal = Calibrater(image, img_size=size[-2::-1], width=tennis_width, height=tennis_height, data_path=None)
    #     mat = cal._get_trans_matrix()
    #     numpy.savetxt('mat_1.csv', mat, delimiter = ',')
    
    my_matrix = numpy.loadtxt(open("mat.csv","rb"),delimiter=",",skiprows=0) 

###################################################################################
    target_json = "/alphapose/alphapose-results.json"
    path = origin + target_json
    test = read_json.read_json(path)

    img_final_list = find_athlete(test, my_matrix, tennis_width, tennis_height)

    img = fn.vis_frame(image,test)

    cv2.imshow('img', img)
    cv2.waitKey(0)

###########################################################
# resize target image. Don't care this part:
    court = cv2.imread("court|.png")
    # court = court[140:705, 160:390, :] # range of height and width
    court = court[140:705, 164:387, :]
    scale = (tennis_width, tennis_height)
    court = cv2.resize(court, scale)

###########################################################
    # cv2.circle(court, people_transf_position, 3, (0,0,0),10)
    for i in range( len(img_final_list) ):
            cv2.circle(court, img_final_list[i].int().tuple(), 3, (0,0,0),10)
    cv2.imshow('court',court)
    cv2.waitKey(0)
###########################################################

