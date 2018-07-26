import pandas as pd
import numpy as np
import curves
import augment
from keras.models import Sequential, load_model
from keras.layers import BatchNormalization, Conv2D, Activation, MaxPooling2D, Dense, Flatten, Dropout
from keras import optimizers

# creating a data-frame to load the training.csv data
print "[INFO] Reading training.csv"
df = pd.read_csv("data/training.csv")

# dropping rows with incomplete data
df2 = df.dropna()

# creating a list that contains all the images but as strings
imageStringList = []
imageStringList.extend(df2.loc[:,"Image"])

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
X1 = X_temp.astype(np.float)[:,:,:,np.newaxis] 				# X1.shape = (2140,96,96,1)
X2 = augment.invert(X1)                                     # X2.shape = (2140,96,96,1)
X = np.concatenate((X1,X2),axis=0)                          # X.shape = (4280,96,96,1)

# output
Y1 = np.stack(df2.ix[:,range(0,30)].values,axis=0) 	        # Y1.shape(2140,30)
Y2 = augment.swap(Y1)                                       # Y2.shape(2140,30)
Y = np.concatenate((Y1,Y2),axis=0)                          # Y.shape(2140,30)

# scaling input range to [0,1] from [0,255]
X = X/255

# scaling target values to [-1,1] from [0,95]
Y = (Y - 48)/48

# creating the ann model
model = Sequential()

model.add(BatchNormalization(input_shape=(96,96,1)))

model.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.13))

model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.13))

model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.13))

model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.13))

model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.13))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.33))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.33))
model.add(Dense(256))
model.add(Dropout(0.33))
model.add(Dense(30))

model.summary()

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

print "[INFO] Training"
hist = model.fit(X, Y, validation_split=0.2, shuffle=True,
 epochs=30, batch_size=32)

model.save('src/cnn50.h5')

# plot loss and metric curves
curves.generate(hist.history,'cnn50')
