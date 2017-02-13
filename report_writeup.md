##Vehicle Detection Project Report

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/ycrcb_hog.png
[image2]: ./output_images/detected_car.png
[image3]: ./output_images/detected_car_heatmap.png

###Here I will consider the [Rubric Points](https://review.udacity.com/#!/rubrics/513/view) individually and describe how I addressed each point in my implementation.  

---

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 51 through 99 of the file called `detection_lib.py`.  

First I read in all the `vehicle` and `non-vehicle` images. I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image1]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of `skimage.hog()` parameters and color space, and found that HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` with `YCrCb` color space seems that best. And it turns out works well in the following task.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in 5th cell of the file called `vehicle_detection.ipynb`.  

First I extract the HOG features with the color space and parameters described above, combined with spatially binned color and histograms of color. Then I stack these features into a single numpy array in the order of spatiall binned color, color histogram and HOG features. After that, I split up data into randomized training and test sets with sklearn `train_test_split` function to suffle the data and aviod overfitting. Finally I train a linear SVM classifier on the trainning datasets, and test on the test datasets with a 99.19% accuracy.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

First of all, I only apply the sliding window on the bottom half of the image because the vehicles will not show above the road surface. I tried different window size (search scale) and overlap rate, and found that window size 96 and overlap 0.5 works well. 

Here is the results on the test image.

![alt text][image2]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I test on different combination of the following three kind of features: YCrCb ALL-channel HOG features,spatially binned color and histograms of color. I also played with number of channels, color space, and the HOG parameter. I turns out to use all three kind of features, and have a pretty good test accuracy 99.19% of my classifier and the following test images showed that it works well.  

Here are some example images:

![alt text][image2]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a link to my video result.

[![IMAGE ALT TEXT](http://img.youtube.com/vi/DF7vj5NXV_Y/0.jpg)](https://youtu.be/DF7vj5NXV_Y "fianl project video ")


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for this step is contained in lines 353 through 369 of the file called `detection_lib.py`.

I recorded the positions of the positive detected cars from the trained classifer in each frame of the video. Then a heat map is created a heatmap from the positive detection, by summing up and recorded the heatmap of 10 sequential frame and apply a threshold of 5, I get a new heat map which is based on the detection of last 10 frame. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  we assumed each blob corresponded to a vehicle, and finally we draw bounding boxes to cover the area of each blob detected.  

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problem with the fianl video includs: several false positive, the bounding box are a little wobbly evern after sum up and smooth the heat map, and the bounding box tends to merger into one if the two cara are close enough. To make it more robust, I need test more combination of the features and test on more classifiers, and most importantly try different scale ans size of the sliding window. 

