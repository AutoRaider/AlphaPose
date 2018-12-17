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
    # #people_position = ge.Point(394,472)
    # target_path = "/alphapose/demo.png"
    # target_img = origin + target_path
    # image = cv2.imread(target_img)

    # size = image.shape # height and width
    # print('origin img size:', size)
    tennis_width = 300
    tennis_height = 400
    scale = (tennis_width, tennis_height)

    
###################################################################################

    # flag = int(input("If you have the transform matrix? \nYes: press '1', No:press '2'\n"))
    # if flag == 2:
    #     cal = Calibrater(image, img_size=size[-2::-1], width=tennis_width, height=tennis_height, data_path=None)
    #     mat = cal._get_trans_matrix()
    #     numpy.savetxt('mat_demo.csv', mat, delimiter = ',')

###################################################################################

    my_matrix = numpy.loadtxt(open("mat_demo.csv","rb"),delimiter=",",skiprows=0) 

###################################################################################
    target_json = "/alphapose/alphapose-results.json"
    path = origin + target_json
    test = read_json.read_json(path)
###################################################################################
    video_path = os.path.join(origin, '/alphapose/badminton.mp4')
    device = cv2.VideoCapture(video_path)
    size = ((int(device.get(cv2.CAP_PROP_FRAME_WIDTH)),int(device.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    success, frame = device.read()
###################################################################################
    success = True
    frame_id = 0
    while success:
        success, frame = device.read()
        img_final_list = find_athlete(test[frame_id], my_matrix, tennis_width, tennis_height)
        # frame_queue.put(frame)
        frame_id = frame_id + 1
        


    img_final_list = find_athlete(test[150], my_matrix, tennis_width, tennis_height)

    img = fn.vis_frame(image,test[150])

    # cv2.imshow('img', img)
    # cv2.waitKey(0)


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
###########################################################
    print(size[0])

    # img[size[0]-scale[1]:size[0], size[1]-scale[0]:size[1], :] = court
    img[size[1]-scale[1]:size[1], size[0]-scale[0]:size[0], :] = court
    img = cv2.resize(img, (0, 0), fx=0.8, fy=0.8, interpolation=cv2.INTER_NEAREST)
    print(':::::::',img.shape)

###########################################################
    cv2.imshow('court',img)
    cv2.waitKey(0)
###########################################################

