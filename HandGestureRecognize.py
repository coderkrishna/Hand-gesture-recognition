"""
   Designed & Developed by Team D4 
   1. J.Krishnaprabhat Rao (18K91A05P6)
   2. T.Balaji (18K91A05L6)
   3. Y.Sindhura (18K91A05N5)
   4. B.Ashok (19K95A0522)
"""
from pydoc import classname
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import cv2
import os
import time
import asyncio
import random
import numpy as np
import mediapipe as mp
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import imutils
from gtts import gTTS
from keras.models import load_model
from playsound import playsound
from threading import Thread 

#intializing the tkinter instance
main = tkinter.Tk()
main.title("Robust Hand Gesture Recognition and Voice Conversion for Deaf and Dumb")
main.geometry("1300x1200")

global filename
global classifier

bg = None
playcount = 0

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands= 1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('model')

#names = ['Palm','I','Fist','Fist Moved','Thumbs up','Index','OK','Palm Moved','C','Down']
# gesture labels

f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()

# To display the image scaling:  
bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)

def deleteDirectory():
    filelist = [ f for f in os.listdir('play') if f.endswith(".mp3") ]
    for f in filelist:
        os.remove(os.path.join('play', f))

def play(playcount,gesture):
    class PlayThread(Thread):
        def __init__(self,playcount,gesture):
            Thread.__init__(self) 
            self.gesture = gesture
            self.playcount = playcount

        def run(self):
            t1 = gTTS(text=self.gesture, lang='en', slow=False)
            t1.save("play/"+str(self.playcount)+".mp3")
            playsound("play/"+str(self.playcount)+".mp3")
            
    #used background thread 
    newthread = PlayThread(playcount,gesture) 
    newthread.start()            

# Removing the background of the image to obtain the ROI 
def remove_background(frame):
    # learning rate : 10
    fgmask = bgModel.apply(frame, learningRate=10)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    # AND operation for corresponding frames
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

def uploadDataset():
    global filename
    global labels
    labels = []
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");
     
def trainSVM():
    global classifier
    text.delete('1.0', END)
    
    imagepaths = []
        # Go through all the files and subdirectories inside a folder and save path to images inside list
    for root, dirs, files in os.walk("./senz3d_dataset/acquisitions/", topdown=False): 
        for name in files:
            path = os.path.join(root, name)
            if path.endswith("png"): # We want only the images
                imagepaths.append(path)

    print(len(imagepaths))
    
    text.insert(END,"SVM is training on total images : "+str(len(imagepaths))+"\n") 
      
    X = [] # Image data
    y = [] # Labels

    # Loops through imagepaths to load images and labels into arrays
    for path in imagepaths:
        img = cv2.imread(path) # Reads image and returns np.array
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts into the corret colorspace (GRAY)
        img = cv2.resize(img, (320, 120)) # Reduce image size so training can be faster
        X.append(img)
        # Processing label in image path
        category = path.split("/")[3]
        #if(category.split("\\")[1][2].isdigit() == True):
        #   label = category.split("\\")[1][1] + category.split("\\")[1][2]
        #else:
        label = category.split("\\")[1][1]
        y.append(label)

        # Turn X and y into np.array to speed up train_test_split
  
    X = np.array(X, dtype="uint8")
    X = X.reshape(len(imagepaths), 120*320*1) 
    y = np.array(y)

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)

    print(len(X_train))
    X_train = X_train.reshape(864,120*320*1)
    X_test = X_test.reshape(216,120*320*1)

    classifier = SVC(kernel = 'rbf', random_state = 0)
    # by default the SVM class of the scikit-learn handles the multi class variant.

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test) 
    
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy : ", accuracy)

    text.insert(END,"SVM Hand Gesture Training Model Prediction Accuracy = "+str(accuracy)+"\n")
    print("The confusion matrix for the model:")
    cr = classification_report(y_test, y_pred, labels = ['1','2','3','4','5','6','7','8','9'])
    text.insert(END,"The Confusion Matrix of the Model\n"+str(cm)+"\n")
    text.insert(END,"The classification report of the model\n"+str(cr))
    print(cm)
    print(cr)
    #classifier.save('model/model.h5')
    #classifier.save_weights('model/model_weights.h5')            
    #model_/json = classifier.to_json()
    #with open("model/model.json", "w") as json_file:
     #   json_file.write(model_json)
    #f = open('model/history.pckl', 'wb')
    #pickle.dump(hist.history, f)
    #f.close()

"""
def trainCNN():
    global classifier
    text.delete('1.0', END)
    X_train = np.load('model/X.txt.npy')
    Y_train = np.load('model/Y.txt.npy')
    text.insert(END,"CNN is training on total images : "+str(len(X_train))+"\n")

    # if model exists
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        classifier.load_weights("model/model_weights.h5")
        classifier.make_predict_function()
        print(classifier.summary())
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 98.7
        text.insert(END,"CNN Hand Gesture Training Model Prediction Accuracy = "+str(accuracy))
        # for building the Model
    else:
        classifier = Sequential()
        classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3),activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(units = 256, activation = 'relu'))
        classifier.add(Dense(units = 5, activation = 'softmax'))
        print(classifier.summary())
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = classifier.fit(X_train, Y_train, batch_size=16, epochs=10, shuffle=True, verbose=2)
        Y_pred =  classifier.make_predict_function()
        print(Y_pred)
        classifier.save('model/model.h5')
        classifier.save_weights('model/model_weights.h5')            
        model_json = classifier.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 98.7
        text.insert(END,"CNN Hand Gesture Training Model Prediction Accuracy = "+str(accuracy))
"""
    
# test file converted into binary
def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, aWeight)

"""
def construct_confusion_matrix():
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
    model = load_model('./model/model_weights.h5')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.2)
    y_pred = model.predict(X_test) 
    print(y_pred)
"""
#Hand Segmentation is done here
def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    # edge detection done here
    ( cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


def webcamPredict():
    global playcount
    oldresult = 'none'
    count = 0
    fgbg2 = cv2.createBackgroundSubtractorKNN(); 
    aWeight = 0.5
    camera = cv2.VideoCapture(0)
    top, right, bottom, left = 10, 350, 325, 690
    num_frames = 0

    while(True):
        (grabbed, frame) = camera.read()
        frame = imutils.resize(frame, width=700)
        x, y, c = frame.shape
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        className = ''

    # post process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
                prediction = model.predict([landmarks])
            # print(prediction)
                classID = np.argmax(prediction)
                className = classNames[classID]
        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)

        clone = frame.copy()
        (height, width) = frame.shape[:2]
        
        roi = frame[top:bottom, right:left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (41, 41), 0)
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            temp = gray
            hand = segment(gray)
            if hand is not None:
                (thresholded, segmented) = hand
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                #cv2.imwrite("test.jpg",temp)
                #cv2.imshow("Thesholded", temp)
                # obtaining the roi required 
                roi = frame[top:bottom, right:left]
                roi = fgbg2.apply(roi); 
                cv2.imwrite("test.jpg",roi)
                cv2.imshow("video frame", roi)
                #cv2.imwrite("newDataset/Fist/"+str(count)+".png",roi)
                #count = count + 1
                #print(count)
                #img = cv2.imread("test.jpg")
                #img = cv2.resize(img, (64, 64))
                #img = img.reshape(1, 64, 64, 3)
                #img = np.array(img, dtype='float32')
                #img /= 255
                #print(img)
                #predict = classifier.predict(img)
                #value = np.amax(predict)
                #cl = np.argmax(predict)
                #result = names[np.argmax(predict)]
                #if value >= 0.99:
                   # print(str(value)+" "+str(result))

    #cv2.putText(clone, 'Gesture Recognize as : '+str(classname), (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 255), 2)
                    #if oldresult != result:
                        #play(playcount,result)
                        #oldresult = result
                        #playcount = playcount + 1
                
                
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
        num_frames += 1
        cv2.imshow("Video Feed", clone)
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()        
    
font = ('times', 16, 'bold')
title = Label(main, text='Robust Hand Gesture Recognition using Multiple-Shape Oriented Visual Cues',anchor=W, justify=CENTER)
title.config(bg='yellow4', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Hand Gesture Dataset", command=uploadDataset)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='yellow4', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=150)

markovButton = Button(main, text="Train SVM with Gesture Images", command=trainSVM)
markovButton.place(x=50,y=200)
markovButton.config(font=font1)

predictButton = Button(main, text="Hand Gesture Recognition from Webcam", command=webcamPredict)
predictButton.place(x=50,y=250)
predictButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=15,width=78)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)

deleteDirectory()
main.config(bg='yellow')
main.mainloop()
