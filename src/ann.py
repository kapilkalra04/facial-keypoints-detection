import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Activation, MaxPooling2D, Dense, GlobalAveragePooling2D, Dropout
from keras import optimizers


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
		print "[INFO] " + str(index) + " images converted"

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

# creating the ann
model = Sequential()

model.add(Dense(units=72, activation='relu', input_shape=(96*96,)))
model.add(Activation('relu'))
model.add(Dense(8))

model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mae'])

print "[INFO] Training"
hist = model.fit(X.reshape(Y.shape[0], -1), Y,
                  validation_split=0.2, shuffle=True,
                  epochs=20, batch_size=1)


# plotting the loss and metric for the model
ax1 = plt.subplot(2,1,1)
plt.plot(hist.history['mean_absolute_error'], linestyle='--', marker='o', color='b')
plt.plot(hist.history['val_mean_absolute_error'], linestyle='--', marker='o', color='g')
level1 = hist.history['mean_absolute_error'][0]
level2 = hist.history['mean_absolute_error'][0] - 0.01
i = 0
for e in hist.history['mean_absolute_error']:
    if i%5==4 :
        e = round(e,4)
        ax1.annotate('('+str(i)+','+str(e)+')', xy=(i,level1), textcoords='data',color='b')
    i = i+1
i = 0
for e in hist.history['val_mean_absolute_error']:
    if i%5==4 :
        e = round(e,4)
        ax1.annotate('('+str(i)+','+str(e)+')', xy=(i,level2), textcoords='data',color='g')
    i = i+1
plt.title('MAE | mean absolute error')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower left')

ax2 = plt.subplot(2,1,2)
plt.plot(hist.history['loss'], linestyle='--', marker='o', color='b')
plt.plot(hist.history['val_loss'], linestyle='--', marker='o', color='g')
level1 = hist.history['loss'][0]
level2 = hist.history['loss'][0] - 0.01
i = 0
for e in hist.history['loss']:
    if i%5==4 :
        e = round(e,4)
        ax2.annotate('('+str(i)+','+str(e)+')', xy=(i,level1), textcoords='data',color='b')
    i = i+1
i = 0
for e in hist.history['val_loss']:
    if i%5==4 :
        e = round(e,4)
        ax2.annotate('('+str(i)+','+str(e)+')', xy=(i,level2), textcoords='data',color='g')
    i = i+1
plt.title('MSE | mean squared error')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower left')

# exporting the plot
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches((16,9), forward=False)
plt.savefig('results/ann.png',dpi=300)
plt.close()
