import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# creating a data-frame to load the training.csv data
print "[INFO] Reading training.csv"
df = pd.read_csv("data/training.csv")

# creating a list that contains all the images
imagesString = []
imagesString.extend(df.loc[: , "Image"])

# converting strings to images
print "[INFO] Converting string enteries to images"
images = []
for s in imagesString:
	images.append(np.array([int (e) for e in s.split(" ")]).reshape((96,96)))

print type(images[0])
plt.imshow(images[0], cmap="gray")
plt.show()
