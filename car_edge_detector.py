import numpy as np
from skimage.feature import hog
import cv2
import matplotlib.pyplot as plt
import random
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import glob
import pickle

pix = 8
cell = 2
orient = 9
col_space = 'YCrCb'

def get_hog(img):
	'''
	converts an image to its HOG version
	'''
	features, hog_img = hog(img, orientations=orient, pixels_per_cell=(pix, pix),
					   cells_per_block=(cell, cell), transform_sqrt=True, 
					   visualize=True, feature_vector=True, block_norm='L1')
	return features, hog_img

def get_feats(files):
	'''
	returns a list of HOG features for multiple files
	'''
	all_feats = []
	for i,file in enumerate(files):
		image = mpimg.imread(file)
		feature_image = np.copy(image)
		hog_features = []
		for channel in range(feature_image.shape[2]):
			hog_features.append(get_hog(feature_image[:,:,channel])[0])
		hog_features = np.ravel(hog_features)
		all_feats.append(hog_features)
		if i%100 == 0 and i>0:
			print(i, 'done')
 
	return all_feats

#training files
vehicle_image_filenames = glob.glob('training_images/vehicles/**/*.png', recursive=True)
non_vehicle_image_filenames = glob.glob('training_images/non-vehicles/**/*.png', recursive=True)

print("Total number of training vehicle images: " + str(len(vehicle_image_filenames)))
print("Total number of training non-vehicle images: " + str(len(non_vehicle_image_filenames)))


pickled = True #needed for re-running the code

vehicle_hog_feats = None
non_vehicle_hog_feats = None


if not pickled:
	#for first run, calculate features
	vehicle_hog_feats = get_feats(vehicle_image_filenames)
	non_vehicle_hog_feats = get_feats(non_vehicle_image_filenames)

	#write features to file
	file_vehicle = open('vehicle', 'wb')
	file_non_vehicle = open('non_vehicle','wb')
	
	pickle.dump(vehicle_hog_feats, file_vehicle)
	pickle.dump(non_vehicle_hog_feats, file_non_vehicle)
	
	file_vehicle.close()
	file_non_vehicle.close()

else:
	#2nd run onwards, read features from file
	file_vehicle = open('vehicle', 'rb')
	file_non_vehicle = open('non_vehicle','rb')

	vehicle_hog_feats = pickle.load(file_vehicle)
	non_vehicle_hog_feats = pickle.load(file_non_vehicle)
	
	file_vehicle.close()
	file_non_vehicle.close()


#scale features X
X = np.vstack((vehicle_hog_feats, non_vehicle_hog_feats)).astype(np.float32)
X_scaler = StandardScaler().fit(X)
X = X_scaler.transform(X)

#crate ground truth labels
y = np.hstack((np.ones(len(vehicle_hog_feats)), np.zeros(len(non_vehicle_hog_feats))))

#80-20 train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = None
if not pickled:
	#for first run, fit the classifier
	clf = LinearSVC()

	clf.fit(X_train, y_train)
	
	print('Model trained')

	#store to file
	file_clf = open('model', 'wb')
	pickle.dump(clf, file_clf)
	file_clf.close()

else:
	#secnd run onwards, load trained model from file
	file_clf = open('model', 'rb')
	clf = pickle.load(file_clf)
	file_clf.close()

#checking on test set
y_pred = clf.predict(X_test)
#print accuracy
print(((y_pred == y_test).astype(int)).mean())

#running on hold out set
test_files = glob.glob('test_images/*')
test_feats = get_feats(test_files)

preds = clf.predict(test_feats)
print(preds)

#print edges using Canny Algorithm for images detected as cars
for index, f in enumerate(test_files):

	if preds[index] == 0:
		#not a car
		print('WARNING MESSAGE!!!')
		continue
	
	img = cv2.imread(f,0)
	edges = cv2.Canny(img,100,275)
	plt.subplot(121),plt.imshow(img,cmap = 'gray')
	plt.title(str(f)), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(edges,cmap = 'gray')
	plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

	plt.show()