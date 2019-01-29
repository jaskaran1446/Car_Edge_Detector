This project is a classifier and edge detection for cars. It uses a Linear SVM classifier for detecting cars, then uses HOG features to obtain its edges. 
If there is no car detected, no edges are displayed and a warning is flashed.

A pickled model is attached for a quick run.

To run the script:
```python car_edge_detector.py```

To download training dataset:
Positive samples(8k+): https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip
Negative samples(8k+): https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip

(Source: Udacity)
