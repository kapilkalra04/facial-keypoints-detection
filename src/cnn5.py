import pandas as pd
import numpy as np
import curves
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Activation, MaxPooling2D, Dense, Flatten, Dropout
from keras import optimizers

# ###
# MODEL INSPIRED FROM
# 1. http://cs231n.stanford.edu/reports/2016/pdfs/007_Report.pdf
# ###

# creating a data-frame to load the training.csv data
print "[INFO] Reading training.csv"
df = pd.read_csv("data/training.csv")

# extracting a dataframe with only important facial landmarks
df2 = df.ix[:,['left_eye_center_x','left_eye_center_y','right_eye_center_x',
'right_eye_center_y','nose_tip_x','nose_tip_y','mouth_center_bottom_lip_x',
'mouth_center_bottom_lip_y','Image']]

# dropping rows with incomplete data
df3 = df2.dropna() 							# total no of rows in df3 = 7000 

# creating a list that contains all the images but as strings
imageStringList = []
imageStringList.extend(df3.loc[:,"Image"])

# converting the strings to numpy arrays
print "[INFO] Converting string enteries to images"
imageList = []
index=0
for s in imageStringList:
	imageList.append(np.array([int (e) for e in s.split(" ")]).reshape((96,96)))
	index = index + 1
	if index%500 == 0 :
		print str(index) + " images converted"

# preparing the training data
# input
X_temp = np.stack(imageList,axis=0)
X = X_temp.astype(np.float)[:,:,:,np.newaxis] 				# X.shape = (7000,96,96,1)

# output
Y = np.stack(df3.ix[:,[0,1,2,3,4,5,6,7]].values,axis=0) 	# Y.shape(7000,8)

# scaling input range to [0,1] from [0,255]
X = X/255

# scaling target values to [-1,1] from [0,95]
Y = (Y - 48)/48

# creating the ann model
model = Sequential()

model.add(BatchNormalization(input_shape=(96,96,1)))                                                # OUTPUT (96,96,1)

model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), kernel_initializer='he_normal'))     # OUTPUT (94,94,32)
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))                                             # OUTPUT (47,47,32)
model.add(Dropout(0.13))

model.add(Conv2D(filters=64, kernel_size=(2,2), strides=(1,1)))                                     # OUTPUT (46,46,64)
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))                                             # OUTPUT (23,23,64)
model.add(Dropout(0.13))

model.add(Conv2D(filters=128, kernel_size=(2,2), strides=(1,1)))                                    # OUTPUT (22,22,128)
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))                                             # OUTPUT (11,11,128)
model.add(Dropout(0.13))

model.add(Flatten());                                                                               # OUTPUT (11*11*128)

model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(8))

model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mae'])

print "[INFO] Training"
hist = model.fit(X, Y, validation_split=0.2, shuffle=True,
 epochs=30, batch_size=1)

# plot loss and metric curves
curves.generate(hist.history,'cnn5')
