import pandas as pd
import numpy as np
import curves
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Activation, MaxPooling2D, Dense, GlobalAveragePooling2D, Dropout
from keras import optimizers

# ###
# MODEL INSPIRED FROM
# 1. http://flothesof.github.io/convnet-face-keypoint-detection.html
# 2. https://towardsdatascience.com/detecting-facial-features-using-deep-learning-2e23c8660a7a
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

model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(1,1), kernel_initializer='he_normal'))    # OUTPUT (92,92,24)
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))                                             # OUTPUT (46,46,24)
model.add(Dropout(0.13))

model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(1,1)))                                    # OUTPUT (42,42,864)
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))                                             # OUTPUT (21,21,864)
model.add(Dropout(0.13))

model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(1,1)))                                    # OUTPUT (17,17,41472)
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))                                             # OUTPUT (9,9,41472)
model.add(Dropout(0.13))

model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1)))                                    # OUTPUT (7,7,2654208)
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))                                             # OUTPUT (4,4,2654208)
model.add(Dropout(0.13))

model.add(GlobalAveragePooling2D());                                                                # OUTPUT (2654208)

model.add(Dense(240, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(8))

model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mae'])

print "[INFO] Training"
hist = model.fit(X, Y, validation_split=0.2, shuffle=True,
 epochs=30, batch_size=1)

# plot loss and metric curves
curves.generate(hist.history,'cnn1')
