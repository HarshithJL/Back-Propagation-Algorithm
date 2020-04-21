import numpy as np
import random
import os
import cv2
from sklearn import preprocessing
import pandas as pd

def oneHotEncoding():
    le = preprocessing.LabelEncoder()
    label_encoded = le.fit_transform(classes)
    onehot_encode = preprocessing.OneHotEncoder(sparse = False)
    label_encoded = label_encoded.reshape( len(label_encoded),1)
    onehot_encode = onehot_encode.fit_transform(label_encoded)

    return onehot_encode

def get_key1(val):
	for key,value in labels.items():
		if key == val:
			return value

def get_key2(val):
	for key,value in labels.items():
		if value == val:
			return key

def activationFn(net):
    return (1/(1+np.exp(-net)))

def feedForward(x,network):

    for i in range(hiddenLayers+1):
        temp = network[i]
        newX =[]
        for j in temp:
            netInput = j[-1:] #bias
            netInput+=np.dot(np.transpose(x),j[:-1])
            # while( netInput >= 255):
            #     netInput = netInput/255 #normalization
            output = activationFn(netInput)
            newX.append(output)
        x = newX

    return x

def load_model():
    try:
        network = {}
        w1 = pd.read_csv('x.csv')
        w1 = np.array(w1)
        network[0] = w1
        w2 = pd.read_csv('y.csv')
        w2 = np.array(w2)
        network[1] = w2
        print("Model loaded successfully..!!")
    except:
        print("Error loading model...")

    return network

def Test():

    sum1  = 0
    sum2 = 0
    for i in classes:
        hit = 0
        images = []
        network = load_model()
        for f in os.listdir(r'C:\Users\LENOVO\harshith\programs\ML\Neural net project\fruits-360_dataset\fruits-360\Test'+"/"+i):
            img = cv2.imread(r'C:\Users\LENOVO\harshith\programs\ML\Neural net project\fruits-360_dataset\fruits-360\Test'+'/'+i+'/'+f)
            resize = cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resize,cv2.COLOR_BGR2GRAY)
            images.append([gray,get_key1(i)])
        sum2 += len(images)
        for j in images:
             input = np.divide(np.reshape(j[0],-1),255)
             outputLayer = feedForward(np.transpose(input),network)
             h1 = outputLayer.index(max(outputLayer))
             expected = j[1]
             h2 = expected.index(max(expected))
             if h1==h2:
                 hit+=1
        sum1 +=hit
        print(i)
        print('No of test images : ',len(images))
        print('No of images correctly recognised : ',hit)
        print('Accuracy : ',(hit/len(images))*100,'%')
        print('\n')

    print('Total test images : ',sum2)
    print('Total images correctly recognised : ',sum1)
    print('Overall accuaracy : ',(sum1/sum2)*100,'%')

    # individual classes
    # check = 'Mango'
    # images = []
    # for f in os.listdir(r'C:\Users\LENOVO\harshith\programs\ML\Neural net project\fruits-360_dataset\fruits-360\Test'+"/"+check):
    #     img = cv2.imread(r'C:\Users\LENOVO\harshith\programs\ML\Neural net project\fruits-360_dataset\fruits-360\Test'+'/'+check+'/'+f)
    #     resize = cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_AREA)
    #     gray= cv2.cvtColor(resize,cv2.COLOR_BGR2GRAY)
    #     images.append(gray)
    #
    # network = load_model()
    # hit = 0
    # for j in images:
    #     input = np.divide(np.reshape(j,-1),255)
    #     outputLayer = feedForward(np.transpose(input),network)
    #     idx = outputLayer.index(max(outputLayer))
    #     lst = [0]*len(classes)
    #     lst[idx] = 1
    #     name = get_key2(lst)
    #     print(name)
    #     if (name == check):
    #         hit+=1
    #
    # print('No of test images : ',len(images))
    # print('No of images correctly recognised : ',hit)
    # print('Accuracy : ',(hit/len(images))*100,'%')

hiddenLayers = 1
img_size = 16
classes = ['Mango','Pomegranate','Orange','Guava']
# classes = ['Guava','Pear']

t = oneHotEncoding()

labels = {}

for x,y in zip(classes,t):
    l = []
    for z in y:
        l.append(z)
    labels[x] = l

Test()
