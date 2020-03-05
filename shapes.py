import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
from keras.models import model_from_json
import imutils
import argparse
import os
import math
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Flatten, Dropout
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import model_from_json
import random
import pickle
import pandas as pd


def detectShape(c):
    shape = 'unknown'
    # calculate perimeter using
    peri = cv2.arcLength(c, True)
    # apply contour approximation and store the result in vertices
    vertices = cv2.approxPolyDP(c, 0.04 * peri, True)

    # If the shape it triangle, it will have 3 vertices
    if len(vertices) == 3:
        shape = 'triangle'

    # if the shape has 4 vertices, it is either a square or
    # a rectangle
    elif len(vertices) == 4:
        # using the boundingRect method calculate the width and height
        # of enclosing rectange and then calculte aspect ratio

        x, y, width, height = cv2.boundingRect(vertices)
        aspectRatio = float(width) / height

        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
            shape = "square"
        else:
            shape = "rectangle"

    # if the shape is a pentagon, it will have 5 vertices
    elif len(vertices) == 5:
        shape = "pentagon"

    # otherwise, we assume the shape is a circle
    else:
        shape = "circle"

    # return the name of the shape
    return shape

### Preprocess image
def constrastLimit(image):
    img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(img_hist_equalized)
    channels[0] = cv2.equalizeHist(channels[0])
    img_hist_equalized = cv2.merge(channels)
    img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
    return img_hist_equalized

def LaplacianOfGaussian(image):
    LoG_image = cv2.GaussianBlur(image, (3,3), 0)           # paramter 
    gray = cv2.cvtColor( LoG_image, cv2.COLOR_BGR2GRAY)
    LoG_image = cv2.Laplacian( gray, cv2.CV_8U,3,3,2)       # parameter
    LoG_image = cv2.convertScaleAbs(LoG_image)
    return LoG_image
    
def binarization(image):
    thresh = cv2.threshold(image,32,255,cv2.THRESH_BINARY)[1]
    #thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    return thresh

def preprocess_image(image):
    image = constrastLimit(image)
    image = LaplacianOfGaussian(image)
    image = binarization(image)
    return image
def load_model():
    from keras.callbacks import LearningRateScheduler, ModelCheckpoint
    np.random.seed(0)
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print(loaded_model.summary())
    #score = loaded_model.evaluate(X_test, y_test, verbose=0)
    #print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
  
    return loaded_model
 
# Find Signs
def removeSmallComponents(image, threshold):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    img2 = np.zeros((output.shape),dtype = np.uint8)
    #for every component in the image, you keep it only if it's above threshold
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    return img2

def findContour(image):
    #find contours in the thresholded image
     contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE    )
    #cnts = cnts[0] if imutils.is_cv2() else cnts[1]
     return contours

def contourIsSign(perimeter, centroid, threshold):
    #  perimeter, centroid, threshold
    # # Compute signature of contour
    result=[]
    for p in perimeter:
        p = p[0]
        distance = sqrt((p[0] - centroid[0])**2 + (p[1] - centroid[1])**2)
        result.append(distance)
    max_value = max(result)
    signature = [float(dist) / max_value for dist in result ]
    # Check signature of contour.
    temp = sum((1 - s) for s in signature)
    temp = temp / len(signature)
    if temp < threshold: # is  the sign
        return True, max_value + 2
    else:                 # is not the sign
        return False, max_value + 2

#crop sign 
def cropContour(image, center, max_distance):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(center[0] - max_distance), 0])
    bottom = min([int(center[0] + max_distance + 1), height-1])
    left = max([int(center[1] - max_distance), 0])
    right = min([int(center[1] + max_distance+1), width-1])
    #print(left, right, top, bottom)
    return image[left:right, top:bottom]

def cropSign(image, coordinate):
    width = image.shape[1]
    height = image.shape[0]
    cropped_width=coordinate[1][0]-coordinate[0][0]
    cropped_height=coordinate[1][1]-coordinate[0][1]
    top = max([int(coordinate[0][1])-int(0.4*cropped_height), 0])
    bottom = min([int(coordinate[1][1])+int(0.4*cropped_height), height-1])
    left = max([int(coordinate[0][0])-int(0.4*cropped_width), 0])
    right = min([int(coordinate[1][0])+int(0.4*cropped_width), width-1])
    #sign=image[top:177,left:340]
    #print(top,left,bottom,right)
    return image[top:bottom,left:right]


def findLargestSign(image, contours, threshold, distance_theshold):
    max_distance = 0
    coordinate = None
    sign = None

    for c in contours:
        
        M = cv2.moments(np.float32(c))
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_sign, distance = contourIsSign(c, [cX, cY], 1-threshold)
        if is_sign and distance > max_distance and distance > distance_theshold:
            max_distance = distance
            coordinate = np.reshape(c, [-1,2])
            left, top = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis = 0)
            coordinate = [(left-2,top-2),(right+3,bottom+1)]
            print('contour',c)
            #dist = cv2.pointPolygonTest(c,(403,216),True)
            #print('dist',dist)
            mask = np.zeros(image.shape,np.uint8)
            cv2.drawContours(mask,[c],0,(255,255,255),5)
            pixelpoints = np.transpose(np.nonzero(mask))

            print('pixelpoints',pixelpoints.shape)
            cv2.imshow('mask', mask)
#pixelpoints = cv.findNonZero(mask)
            cv2.drawContours(image,c,-1,(0,255,0),3)
            sign = cropSign(mask,coordinate)
            img_shape=detectShape(c)
    return sign, coordinate


def findSigns(image, contours, threshold, distance_theshold):
    signs = []
    coordinates = []
    for c in contours:
        # compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_sign, max_distance = contourIsSign(c, [cX, cY], 1-threshold)
        if is_sign and max_distance > distance_theshold:
            sign = cropContour(image, [cX, cY], max_distance)
            signs.append(sign)
            coordinate = np.reshape(c, [-1,2])
            top, left = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis = 0)
            coordinates.append([(top-2,left-2),(right+1,bottom+1)])
    return signs, coordinates

def localization(image, min_size_components, similitary_contour_with_circle, count, current_shape_type):
    original_image = image.copy()
    binary_image = preprocess_image(image)

    binary_image = removeSmallComponents(binary_image, min_size_components)

    binary_image = cv2.bitwise_and(binary_image,binary_image, mask=remove_other_color(image))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)      #-- HSV conversion
    h, s, v = cv2.split(hsv)

    # cv2.imshow('hsv Image', hsv)
    # cv2.imshow('h Image', h)
    # cv2.imshow('s Image', s)
    # cv2.imshow('v Image', v)

    cv2.imshow('BINARY IMAGE', binary_image)
    contours = findContour(binary_image)
    
    
    #signs, coordinates = findSigns(image, contours, similitary_contour_with_circle, 15)
    sign, coordinate = findLargestSign(original_image, contours, similitary_contour_with_circle, 15)
    if sign is not None:
        imgray = cv2.cvtColor(sign, cv2.COLOR_BGR2GRAY)
        cv2.imshow("imgray", imgray)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(sign, [contours[0]], -1, (0,255,0), -1)
   # cv2.waitKey(1)
    #print(sign,coordinate)
    text = ""
    shape_type = -1
    i = 0

    if sign is not None:
        shape_type = 7
     
        #font = cv2.FONT_HERSHEY_SIMPLEX
       # cv2.putText(image,'test',(10,10), font, 6, (200,255,155), 13, cv2.LINE_AA)
        cv2.imshow("sign", sign)
    
    return coordinate, original_image, shape_type, text,sign

def remove_line(img):
    gray = img.copy()
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    minLineLength = 5
    maxLineGap = 3
    lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength,maxLineGap)
    mask = np.ones(img.shape[:2], dtype="uint8") * 255
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(mask,(x1,y1),(x2,y2),(0,0,0),2)
    return cv2.bitwise_and(img, img, mask=mask)

def remove_other_color(img):
    frame = cv2.GaussianBlur(img, (3,3), 0) 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([100,128,0])
    upper_blue = np.array([215,255,255])
    # Threshold the HSV image to get only blue colors
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_white = np.array([0,0,128], dtype=np.uint8)
    upper_white = np.array([255,255,255], dtype=np.uint8)
    # Threshold the HSV image to get only blue colors
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lower_black = np.array([0,0,0], dtype=np.uint8)
    upper_black = np.array([170,150,50], dtype=np.uint8)

    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    mask_1 = cv2.bitwise_or(mask_blue, mask_white)
    mask = cv2.bitwise_or(mask_1, mask_black)
    # Bitwise-AND mask and original image
    #res = cv2.bitwise_and(frame,frame, mask= mask)
    return mask
    # show the output image
    cv2.imshow("Image", image)
    
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img = cv2.equalizeHist(img)
    return img
def preprocess(img):
    
    img = grayscale(img)
    img=~img
    img = equalize(img)
    img = img/255
    return img
def invert(im):
    im[np.where((im == 0).all(axis = 1))] = 250
    im[np.where((im == 255).all(axis = 1))] = 0
    return im
def getShape(img,model):
    img = np.asarray(img)
    img = cv2.resize(img, (60, 60))
    img = preprocess(img)
    #img=invert(img)
    print('black',img)
    cv2.imshow('Processed sign', img)
    
    #plt.imshow(img, cmap = plt.get_cmap('gray'))
    #print(img.shape)
    img = img.reshape(1, 60, 60, 1)
    
    shapes=['triangle','star','square','circle']
    
    print("Sign shape: "+ str(model.predict_classes(img))+ shapes[int(model.predict_classes(img))])
    shape_type=model.predict_classes(img)
    shape_prediction=model.predict(img)
    
    return shapes[int(model.predict_classes(img))],shape_prediction,shape_type
def main(args):
    shapes=['triangle','star','square','circle']
    # load model
    model=load_model()
	

    vidcap = cv2.VideoCapture(args.file_name)

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = vidcap.get(3)  # float
    height = vidcap.get(4) # float

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, fps , (640,480))

    # initialize the termination criteria for cam shift, indicating
    # a maximum of ten iterations or movement by a least one pixel
    # along with the bounding box of the ROI
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    roiBox = None
    roiHist = None

    success = True
    similitary_contour_with_circle = 0.65   # parameter
    count = 0
    current_sign = None
    current_text = ""
    current_size = 0
    sign_count = 0
    coordinates = []
    position = []
    file = open("Output.txt", "w")
    while True:
        success,frame = vidcap.read()
    
        if not success:
            print("FINISHED")
            break
        width = frame.shape[1]
        height = frame.shape[0]
        #frame = cv2.resize(frame, (640,int(height/(width/640))))
        frame = cv2.resize(frame, (640,480))

        print("Frame:{}".format(count))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        coordinate, image, shape_type, text,sign = localization(frame, args.min_size_components, args.similitary_contour_with_circle, count, current_sign)
        
       # print('coordinates',coordinate)
        if coordinate is not None:
            img_shape,shape_prediction,shape_type=getShape(sign,model)
            cv2.rectangle(image, coordinate[0],coordinate[1], (0, 255, 255), 5)
            #print("Sign:{}".format(shape_type))
            font = cv2.FONT_HERSHEY_COMPLEX
            pos=(coordinate[0][0],coordinate[0][1]-30)
            pos1=(coordinate[0][0],coordinate[0][1]-12)
            # cv2.putText(image,img_shape,pos1, font, 1, (0,255,0), 1, cv2.LINE_AA)
            # cv2.putText(image,str(shape_prediction[0][shape_type[0]]),pos, font, 1, (0,255,0), 1, cv2.LINE_AA)

            bar_length=coordinate[0][0]+(coordinate[1][0]-coordinate[0][0])*shape_prediction[0][shape_type[0]]
            print("predicted sign: "+ str(shape_type))
            print("predictions: ",shape_prediction[0][shape_type[0]])
            cv2.putText(image,shapes[shape_type[0]],pos, font, 0.5, (200,255,155), 1, cv2.LINE_AA)
            cv2.rectangle(image, (coordinate[0][0],coordinate[0][1]-25),(coordinate[1][0],coordinate[0][1]-10), (204, 255, 229), 1)
            cv2.rectangle(image, (coordinate[0][0],coordinate[0][1]-20),(int(bar_length),coordinate[0][1]-15), (204, 255, 229), 10)
            cv2.putText(image,str(shape_prediction[0][shape_type[0]]),pos1, font, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.imshow('Result', image)
        #cv2.waitKey(0)
        
        
        count = count + 1
        #Write to video
        out.write(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    file.write("{}".format(1))
    for pos in coordinates:
        file.write("\n{} {} {} {} {} {}".format(pos[0],pos[1],pos[2],pos[3],pos[4], pos[5]))
    print("Finish {} frames".format(count))
    file.close()
    return 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NLP Assignment Command Line")
    
    parser.add_argument(
      '--file_name',
      default= "./4.jpg",
      help= "Video to be analyzed"
      )
    
    
    parser.add_argument(
      '--min_size_components',
      type = int,
      default= 300,
      help= "Min size component to be reserved"
      )

    
    parser.add_argument(
      '--similitary_contour_with_circle',
      type = float,
      default= 0.65,
      help= "Similitary to a circle"
      )

    parser.add_argument(
      '--img_shape',
      type = str,
      default= "",
      help= "shape of countour"
      )
    
    args = parser.parse_args()
    main(args)



cv2.waitKey(0)
cv2.destroyAllWindows()