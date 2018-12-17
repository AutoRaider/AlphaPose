import cv2
import os
import numpy
import sys
import read_json
origin = os.path.dirname(os.getcwd())
sys.path.insert(0,origin + '/tennis')
import geometry as ge
from calibration import Calibrater

target_path = "/alphapose/badminton.jpg"
target_img = origin + target_path


if __name__ == '__main__':
    people_position = ge.Point(394,472)
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

    print("read origin:", my_matrix)
###################################################################################
    origin = os.path.dirname(os.getcwd())
    target_json = "/alphapose/alphapose-results_t.json"
    path = origin + target_json
    test = read_json.read_json(path)

    print('result:', type(test['result']))

    print( type(test['result'][0]['keypoints'][0]) )

    print('here:      ', test['result'][0]['keypoints'][15])
    print('here:      ', test['result'][0]['keypoints'][16])
    
    print('origin length:',len(test['result']))
    img_final_list = []
    for i in range( len(test['result'])-1, -1, -1 ):
        mean_point = (test['result'][i]['keypoints'][15] + test['result'][i]['keypoints'][16]) / 2
        people_position = ge.Point(int(mean_point[0]), int(mean_point[1]))
        people_transf_position = people_position.perspective(my_matrix)
        x = people_transf_position.x
        y = people_transf_position.y
        if x<0 or x>300 or y<0 or y>400 :
            del test['result'][i]
        else:
            img_final_list.append(people_transf_position)
    print('final length:',len(test['result']))

###################################################################################
    # people_transf_position = people_position.perspective(my_matrix)


    # print('transform position:', people_transf_position.int())
    # people_transf_position = people_transf_position.int().tuple()

###########################################################
# resize target image. Don't care this part:
    court = cv2.imread("court|.png")
    # court = court[140:705, 160:390, :] # range of height and width
    court = court[140:705, 164:387, :]
    scale = (300,400)
    court = cv2.resize(court, scale)
    # <class 'geometry.Point'>
    print(img_final_list[0], img_final_list[1])
    print('++++++++++++++++++:',type(img_final_list[0]))
#(tensor(98.2867, dtype=torch.float64), tensor(327.6532, dtype=torch.float64))
    # print('++++++++++++++++++:', img_final_list[0].tuple())
###########################################################
    # cv2.circle(court, people_transf_position, 3, (0,0,0),10)
    for i in range( len(img_final_list) ):
            # cv2.circle(court, (img_final_list[i].x, img_final_list[i].y), 3, (0,0,0),10)
            cv2.circle(court, img_final_list[i].int().tuple(), 3, (0,0,0),10)
    cv2.imshow('court',court)
    cv2.waitKey()
###########################################################

