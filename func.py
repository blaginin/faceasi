import dlib
from skimage import io, color
import numpy as np
from PIL import Image
from json import dumps, loads
from requests import get as rget
from threading import Thread
import imutils
import cv2
from time import time as get_time
from bs4 import BeautifulSoup
import pandas as pd
from find import find

fcount = 5
pframe = 1
def updateDescriptors(desc, name):
    global savedDescriptors
    desc = list(desc)
    if len(desc) == 128:
        record = {
            "name": name,
            "descriptor": desc
        }
        try:
            savedDescriptors.append(record)
            saveDescriptors(savedDescriptors)
        except AttributeError:
            print("error")


def loadDescriptors():
    with open("asidescriptors.json", 'r') as file:
        savedDescriptors = loads(file.read())
        file.close()
    return savedDescriptors


def saveDescriptors(savedDescriptors):
    with open("asidescriptors.json", 'w') as file:
        file.write(dumps(savedDescriptors, indent=4))
        file.close()


def extract_descriptor(img):
    try:
        face_face = []
        dets_webcam = detector(img, 1)
        for k, d in enumerate(dets_webcam):
            shape = sp(img, d)
            face_face = facerec.compute_face_descriptor(img, shape)
        return face_face
    except RuntimeError:
        print('error')


def face_compare_np(face1, face_arr):
    try:
        return np.linalg.norm(np.asarray(face1)-np.asarray(face_arr))
    except Exception:
        return 100


def get_person_name(img):
    test_face = extract_descriptor(img)
    return find(test_face, savedDescriptors)


def norm_image(img):
    img_y, img_b, img_r = img.convert('YCbCr').split()
    img_y_np = np.asarray(img_y).astype(float)

    img_y_np /= 255
    img_y_np -= img_y_np.mean()
    img_y_np /= img_y_np.std()
    scale = np.max([np.abs(np.percentile(img_y_np, 1.0)),
                    np.abs(np.percentile(img_y_np, 99.0))])
    img_y_np = img_y_np / scale
    img_y_np = np.clip(img_y_np, -1.0, 1.0)
    img_y_np = (img_y_np + 1.0) / 2.0
    img_y_np = (img_y_np * 255 + 0.5).astype(np.uint8)

    img_y = Image.fromarray(img_y_np)
    img_ybr = Image.merge('YCbCr', (img_y, img_b, img_r))
    img_nrm = img_ybr.convert('RGB')

    return img_nrm


def img_parse():
    global savedDescriptors
    print("Starting")
    if len(savedDescriptors) > 0:
        sti = int(savedDescriptors[len(savedDescriptors)-1]["name"]) + 1
    #else:
    #    sti = 60000
    for i in range(sti, 127015):
        page = 'https://leader-id.ru/files/user_photo/{0}/{0}_300.jpg'.format(i)
        r = rget(page, stream=True)
        if r.status_code == 200:
            f = open('img.jpg','wb')
            f.write(r.content)
            f.close()
            img = color.gray2rgb(io.imread(f.name))
            updateDescriptors(extract_descriptor(img), str(i))
            print(i)
        # else:
        #     f = open('people.txt','a')
        #     f.write(str(i) + '\n')
        #     f.close()
    print("Finished")


def video_detect(face_cascade, frame):
    global pframe
    global fcount
    people = []
    frame = imutils.resize(frame)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(20, 20),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            subres = get_person_name(frame[y*4:y*4 + h*4, x*4:x*4 + w*4])
            #if subres is not None:
                #result.append(subres)
                #name = str(get_person_name(frame))
            print(subres)
            cv2.imwrite('static/frame' + str(pframe) + '.png', frame[y*4:y*4 + h*4, x*4:x*4 + w*4])
            if pframe < 10:
                pframe+=1
            else:
                pframe = 1
            people.append(subres)

    #if fcount == 5:
    cv2.imwrite('static/frame.png', frame)
    #fcount = 1
    #else:
    #    fcount+=1
    return people


def camera_detect(face_cascade, detector, facerec, sp, frame):
    result = []
    frame = imutils.resize(frame)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(20, 20),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        x*= 4
        y*= 4
        w*= 4
        h*= 4
        subres = get_person_name(small_frame[y:y+h, x:x+w])
        if subres is not None:
            result.append(subres)
    return result


def name_parse(id):
    if id != None:
        source = rget('https://leader-id.ru/'+id)
        source.encoding = 'utf-8'
        soup = BeautifulSoup(source.text, 'lxml')
        name = soup.find_all('strong')
        if len(name)>0:
            return name[0].text
        else:
            return '0'
    return '0'


def loadDima():
    with open("asidescriptorsvideo.json", 'r') as file:
        dimaDesc = loads(file.read())
        file.close()
    return dimaDesc

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
people = []

try:
    savedDescriptors = loadDescriptors()
    print(savedDescriptors[len(savedDescriptors)-1]["name"])
    #img = io.imread('34761_300.jpg')
    #updateDescriptors(extract_descriptor(img), '34761')
    #print(len(savedDescriptors))
    #npDesc = pd.Series(savedDescriptors)
    #print(1)
    #savedDescriptors = sorted(savedDescriptors, key=lambda k: int(k['name']))
    #print(savedDescriptors[len(savedDescriptors)-1]["name"])
    #dimaDesc = loadDima()
    #print(dimaDesc[1]["name"])
    #savedDescriptors.append(dimaDesc[1])
    #print('here')
    #saveDescriptors(savedDescriptors)
except:
    savedDescriptors = list()
#img = io.imread('C:\\Users\\MSI\\Desktop\\recofi-master(1)\\recofi-master\\static\\dima.jpg')
#updateDescriptors(extract_descriptor(img), '87059')
#img_parse()
#np.unique(npDesc)
#img_parse()
