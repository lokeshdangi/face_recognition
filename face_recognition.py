# remove opencv-python
# pip install opencv-contrib-python

import cv2
import os
import numpy as np
import math
import matplotlib.pyplot as plt

"""
webcam = cv2.VideoCapture(0)
ret, frame = webcam.read()
print (ret)
webcam.release()


cv2.startWindowThread()

cv2.namedWindow("picture window",cv2.WINDOW_NORMAL)
cv2.imshow("picture window",frame)

cv2.waitKey()
cv2.destroyAllWindows()

print(type(frame))

plt.imshow(frame)
plt.show()


frame_RGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
plt.imshow(frame_RGB)
plt.show()

cv2.imwrite('images/img1.jpg',frame)
cv2.imwrite('images/img2.jpg',frame_RGB)

picture_BGR = cv2.imread('images/img1.jpg',1)
picture_RGB = cv2.imread('images/img2.jpg',1)

picture = np.hstack((picture_BGR,picture_RGB))
plt.imshow(picture)



cv2.namedWindow("cam feed",cv2.WINDOW_NORMAL)
webcam = cv2.VideoCapture(0)
while True:
    _,frame = webcam.read()
    cv2.imshow("cam feed",frame)
    
    if cv2.waitKey(20) & 0xFF == 27:
        break

webcam.release()
cv2.destroyAllWindows()


webcam = cv2.VideoCapture(0)
cv2.namedWindow("video",cv2.WINDOW_AUTOSIZE)
fourcc = cv2.VideoWriter_fourcc(*'H264')
video = cv2.VideoWriter('images/video.mp4',fourcc,20.0,(460,480))

while webcam.isOpened():
    ret,frame = webcam.read()
    video.write(frame)
    cv2.imshow('video',frame)
    if cv2.waitKey(40) & 0xFF == 27:
        break

webcam.release()
video.release()
cv2.destroyAllWindows()





face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
webcam = cv2.VideoCapture(0)

def FaceDetect():
    while True:
        ret,frame = webcam.read()
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('img',frame)
        if cv2.waitKey(40) & 0xFF == 27:
            break

webcam.release()
cv2.destroyAllWindows()



"""

def cut_faces(image, faces_coord):
    faces = []
    
    for (x, y, w, h) in faces_coord:
        w_rm = int(0.3 * w / 2)
        faces.append(image[y: y + h, x + w_rm: x + w - w_rm])
         
    return faces

def normalize_intensity(images):
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3 
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images_norm.append(cv2.equalizeHist(image))
    return images_norm

def resize(images, size=(50, 50)):
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size, 
                                    interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, 
                                    interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)

    return images_norm




def normalize_faces(frame, faces_coord):
    faces = cut_faces(frame, faces_coord)
    faces = normalize_intensity(faces)
    faces = resize(faces)
    return faces






"""

webcam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret,frame = webcam.read()
    face_cords = face_cascade.detectMultiScale(frame, 1.3, 5)
    cv2.imshow('full screen',frame)
    if len(face_cords):
        faces = cut_faces(frame,face_cords)
        faces_normalized = normalize_intensity(faces)
        faces_resize = resize(faces_normalized)
        cv2.imshow('face',faces[0])
        cv2.imshow('normalized',faces_normalized[0])
        cv2.imshow('resized',faces_resize[0])

    if cv2.waitKey(40) & 0xFF == 27:
        break


cv2.destroyAllWindows()
webcam.release()


"""



folder = "images/" + input('Person :').lower()
webcam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


if not os.path.exists(folder):
    os.mkdir(folder)
    counter = 0
    timer = 0
    
    while counter < 100:
        ret,frame = webcam.read()
        face_cords = face_cascade.detectMultiScale(frame, 1.3, 5)
        if len(face_cords):
            faces = normalize_faces(frame,face_cords)
            cv2.imwrite(folder + '/' + str(counter) + '.jpg',faces[0])
            cv2.imshow('face',faces[0])
            counter += 1

        timer += 50
        
    cv2.destroyAllWindows()
    webcam.release()

else:
    print("Folder already Exits")
    webcam.release()


#############################################


def collect_dataset():
    images = []
    labels = []
    labels_dic = {}
    people = [person for person in os.listdir("images/")]
    for i, person in enumerate(people):
        labels_dic[i] = person
        for image in os.listdir("images/" + person):
            images.append(cv2.imread("images/" + person + '/' + image, 0))
            labels.append(i)
    return (images, np.array(labels), labels_dic)






images, labels, labels_dic = collect_dataset()

rec_eig = cv2.face.EigenFaceRecognizer_create()
rec_eig.train(images, labels)

# needs at least two people 
rec_fisher = cv2.face.FisherFaceRecognizer_create()
rec_fisher.train(images, labels)

rec_lbph = cv2.face.LBPHFaceRecognizer_create()
rec_lbph.train(images, labels)

print ("Models Trained Succesfully")





   
cv2.namedWindow("cam feed",cv2.WINDOW_NORMAL)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)
while True:
    _,frame = webcam.read()
    cv2.imshow("cam feed",frame)
    face_cords = face_cascade.detectMultiScale(frame, 1.3, 5)
    if len(face_cords):
        faces = normalize_faces(frame,face_cords)
        cv2.imshow('face',faces[0])
        pred = faces[0]
        break
    
    if cv2.waitKey(20) & 0xFF == 27:
        break

webcam.release()
cv2.destroyAllWindows()


prediction,confidence = rec_eig.predict(pred)
print(labels_dic[prediction],", ",confidence)


prediction,confidence = rec_fisher.predict(pred)
print(labels_dic[prediction],", ",confidence)


prediction,confidence = rec_lbph.predict(pred)
print(labels_dic[prediction],", ",confidence)

















