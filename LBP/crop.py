from re import X
import cv2
import os

path='result.txt'
folder = "data/ears/test/"
perfect_ears_folder = 'data/perfectly_detected_ears/test/'

for img in os.listdir(folder):
    im = cv2.imread(os.path.join(folder,img))
    myfile=open(path,'r')
    irofile = iter(myfile)
    for line in myfile:
        name = 'data/test/' + img
        name_line = line.split(":")[0]
        if name_line == name:
            ear_line = next(irofile)
            if ear_line[:3] == "Ear":
                Cord=ear_line.split("(")[1].split(")")[0].split("  ")

                x_min=int(Cord[1])
                x_max=x_min + int(Cord[5])
                y_min=int(Cord[3])
                y_max=y_min+ int(Cord[7])

                crop_img = im[y_min:y_max, x_min:x_max]
                cv2.imwrite("data/ear_detection_predictions/" + img, crop_img)
            else:
                im = cv2.imread(os.path.join(perfect_ears_folder,img))
                cv2.imwrite("data/ear_detection_predictions/" + img, im)
    myfile.close()




