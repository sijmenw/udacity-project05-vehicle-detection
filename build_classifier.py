# created by Sijmen van der Willik
# 06/08/2018 11:17
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split

import lesson_functions


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            else:
                print("COLOR SPACE {} NOT FOUND".format(cspace))
                return False
        else:
            feature_image = np.copy(image)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(lesson_functions.get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = lesson_functions.get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)

        # Apply bin_spatial() to get spatial color features
        spatial_features = lesson_functions.bin_spatial(feature_image, size=(spatial_size, spatial_size))
        # Apply color_hist() also with a color space option now
        hist_features = lesson_functions.color_hist(feature_image, nbins=hist_bins)
        # Append the new feature vectors to the features list
        features.append(np.concatenate((hog_features, spatial_features, hist_features)))

    # Return list of feature vectors
    return features


# Read in car and non-car images
images = glob.glob('images/*/*/*.png')
cars = []
notcars = []
for image in images:
    if '/non-vehicles/' in image:
        notcars.append(image)
    elif '/vehicles/':
        cars.append(image)

print("Number of car images: {}".format(len(cars)))
print("Number of notcar images: {}".format(len(notcars)))

# Reduce the sample size because HOG features are slow to compute
# The quiz evaluator times out after 13s of CPU time
sample_size = -1
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]

print("Sample size: {}".format(sample_size))

# Tweakable parameters
colorspace = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 8
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
# color
spatial_size = 32
hist_bins = 32

t = time.time()
car_features = extract_features(cars, cspace=colorspace, orient=orient,
                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                hog_channel=hog_channel)
notcar_features = extract_features(notcars, cspace=colorspace, orient=orient,
                                   pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                   hog_channel=hog_channel)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to extract HOG features...')

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

print('Using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t = time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these', n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

# build dict and save the classifier and parameters
save_dict = {
    "svc": svc,
    "scaler": X_scaler,
    "orient": orient,
    "pix_per_cell": pix_per_cell,
    "cell_per_block": cell_per_block,
    "spatial_size": spatial_size,
    "hist_bins": hist_bins,
    "colorspace": colorspace
}

pickle.dump(save_dict, open("svc_save.p", "wb"))
print("saved svc to .p file")

# confirm correctly saved
svc2 = pickle.load(open("svc_save.p", "rb"))["svc"]
print('My loaded SVC predicts: ', svc2.predict(X_test[0:n_predict]))
