from urllib.request import urlretrieve
from os.path import isfile
import pickle
import numpy as np
import math
import tensorflow as tf
import csv
from scipy import misc
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Convolution1D
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, MaxPooling2D
from keras.models import Sequential
import json
import os

def preprocess(rbgImgArr):
    preprocessImgArr=[]
    for rbgImg in rbgImgArr:
        processed_img=[]
        for row in rbgImg:
            for elem in row:
                avg = float(0.299*elem[0]+0.587*elem[1]+0.114*elem[2])
                processed_img.append(avg)
        preprocessImgArr.append(processed_img)
    return preprocessImgArr

def get_model(time_len=1):
    row, col, ch = 160, 320, 3  

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(261))
    model.add(Dropout(.3))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

f=open('driving_log.csv')
csv_f = csv.reader(f)
train_raw_data=[]
train_data=[]
train_label=[]
count = 0
for row in csv_f:
    if count!=0:
        train_label.append(row[3])
        train_raw_data.append(misc.imread(row[0]))
    count+=1
print("done!")


f=open('driving_log1.csv')
csv_f = csv.reader(f)
count = 0
for row in csv_f:
    if count!=0:
        train_label.append(row[3])
        train_raw_data.append(misc.imread(row[0]))
    count+=1
print("done!")

f=open('driving_log2.csv')
csv_f = csv.reader(f)
count = 0
for row in csv_f:
    if count!=0:
        train_label.append(row[3])
        train_raw_data.append(misc.imread(row[0]))
    count+=1
print("done!")

f=open('driving_log3.csv')
csv_f = csv.reader(f)
count = 0
for row in csv_f:
    if count!=0:
        train_label.append(row[3])
        train_raw_data.append(misc.imread(row[0]))
    count+=1
print("done!")

#f=open('driving_log4.csv')
#csv_f = csv.reader(f)
#count = 0
#for row in csv_f:
#    if count!=0:
#        train_label.append(row[3])
#        train_raw_data.append(misc.imread(row[0]))
#    count+=1
#print("done!")


f=open('driving_log5.csv')
csv_f = csv.reader(f)
count = 0
for row in csv_f:
    if count!=0:
        train_label.append(row[3])
        train_raw_data.append(misc.imread(row[0]))
    count+=1
print("done!")

f=open('driving_log6.csv')
csv_f = csv.reader(f)
count = 0
for row in csv_f:
    if count!=0:
        train_label.append(row[3])
        train_raw_data.append(misc.imread(row[0]))
    count+=1
print("done!")

f=open('driving_log7.csv')
csv_f = csv.reader(f)
count = 0
for row in csv_f:
    if count!=0:
        train_label.append(row[3])
        train_raw_data.append(misc.imread(row[0]))
    count+=1
print("done!")

f=open('driving_log8.csv')
csv_f = csv.reader(f)
count = 0
for row in csv_f:
    if count!=0:
        train_label.append(row[3])
        train_raw_data.append(misc.imread(row[0]))
    count+=1
print("done!")

f=open('driving_log9.csv')
csv_f = csv.reader(f)
count = 0
for row in csv_f:
    if count!=0:
        train_label.append(row[3])
        train_raw_data.append(misc.imread(row[0]))
    count+=1
print("done!")

f=open('driving_log10.csv')
csv_f = csv.reader(f)
count = 0
for row in csv_f:
    if count!=0:
        train_label.append(row[3])
        train_raw_data.append(misc.imread(row[0]))
    count+=1
print("done!")




train_raw_data=np.array(train_raw_data)

model = Sequential()
model=get_model()
history = model.fit(train_raw_data, train_label, batch_size=30, nb_epoch=3,  validation_split=0.001)

model.save_weights("./model.h5", True)
with open('./model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)

print(model.summary())
