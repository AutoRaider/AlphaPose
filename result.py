import cv2
import os
import numpy
import sys
import read_json
import fn
import copy
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
###########################################################
# resize target image. Don't care this part:
    court = cv2.imread("court|.png")
    # court = court[140:705, 160:390, :] # range of height and width
    court = court[140:705, 164:387, :]
    scale = (tennis_width, tennis_height)
    court = cv2.resize(court, scale)

###################################################################################
    video_path = origin + '/alphapose/badminton.mp4'
    print('video_path:',video_path)
    device = cv2.VideoCapture(video_path)
    size = ((int(device.get(cv2.CAP_PROP_FRAME_WIDTH)),int(device.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1920,1080))
###################################################################################
    success = True
    frame_id = 0
    while success:
        temp_court = copy.deepcopy(court)
        success, frame = device.read()
        print('success:', frame_id)
        img_final_list = find_athlete(test[frame_id], my_matrix, tennis_width, tennis_height)
        # frame_queue.put(frame)
        img = fn.vis_frame(frame, test[frame_id])
        frame_id = frame_id + 1
        if frame_id == len(test):
            break

        for i in range( len(img_final_list) ):
            cv2.circle(temp_court, img_final_list[i].int().tuple(), 3, (0,0,0),10)

        img[size[1]-scale[1]:size[1], size[0]-scale[0]:size[0], :] = temp_court
        out.write(img)
        cv2.destroyAllWindows()

    print('success!')
    device.release()
    out.release()


    # cv2.imshow('img', img)
    # cv2.waitKey(0)

###########################################################


    # img[size[0]-scale[1]:size[0], size[1]-scale[0]:size[1], :] = court
#     img[size[1]-scale[1]:size[1], size[0]-scale[0]:size[0], :] = court
#     img = cv2.resize(img, (0, 0), fx=0.8, fy=0.8, interpolation=cv2.INTER_NEAREST)



