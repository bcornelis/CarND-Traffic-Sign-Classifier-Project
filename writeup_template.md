**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[initial_histogram]: ./histogram_initial.png "Initial Data Histogram Visualization"
[proprocess_grayscale]: ./visualize_preprocess_grayscale.png "Preprocessing: grayscale"
[preprocess_augment_rotate]: ./visualize_preprocess_rotate.png "Preprocessing: rotation"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/bcornelis/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

Data Set Summary & Exploration

1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  
I used default python code to get the length of some arrays, find the shape of the images and numphy.unique to find out how many different items (classes) the data contains,

* The size of training set is: <br>
n_train = len(X_train) = 34799
* The size of test set is:  <br>
n_test = len(X_test) = 12630
* The shape of a traffic sign image is: <br>
image_shape = X_train[0].shape = (32, 32, 3)
* The number of unique classes/labels in the data set is: <br>
n_classes = len(np.unique(y_train)) = 43

2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It's a histogram showing how the sample data is distributed according to the final classes. This kind of visualization directly shows that for some classes, there is a lot of training data (for example class 1 and 2) and for other classes theres very few training samples (for example class 19). The interesting thing about this information is:
* if there's only a very limit set of data for a specific class, there's a (very) low chance the CNN can properly classify objects of this class
* it clearly shows that for additional data (fake data generation, more sample data, ...) we'll have to focus on those classes (with the lowest number of samples) first.

![Initial Data Histogram Visualization][initial_histogram]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook. The two methods in there are <i>applyPreProcessing</i> to normalize and grayscale the images, and <i>generateFakeData</i> to generate more samples.

First all images have been converted to grayscale. The reason for doing this is that when using a color image, the CNN would also learn about colors (and probably after a lot of sample data find out itself) which is irrelevant for traffic sign generation. So by converting the images to grayscale, there's only a single color dimension, instead of 3 which the CNN can learn from.

![Preprocessing: grayscale][proprocess_grayscale]

Next the image is normalized. As the image is grayscale, there's a single color component in the range from [0, 255]. By normalizing the image (divide by 255.) the range is reduced to [0., 1.]

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training, validation and test set was already included, as it was included in the  traffic-signs-data.zip file. The methods used here is the picle library which allows to serialize data to a file, and reload it (deserialize it) afterwards. As there were already 3 test sets included, no extra work was required.

In cell 4, in the generateFakeData method, for augmenting the data set, I decided to generate some random data, starting from the already existing images, and apply simple rotations using cv2.getRotationMatrix2D from the OpenCV library. First I find out what the mean number of samples per class in the test set is. I generate more images for all classes not having this number of images in the set.

Here's a sample augmented image using rotation:

![Preprocess: rotation][preprocess_augment_rotate]

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the eigth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		  |     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscaled and normalized image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, 1 input feature map, 6 output classes, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, 2x2 filter, valid padding, outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, 6 input feature maps, 16 output classes, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, 2x2 filter, valid padding, outputs 5x5x16 				|
| flatten |  5x5x16 input, 400 output
| Fully connected		| input 400, output 120        									|
| RELU					|												|
| Dropout | keep probability 0.6 |
| Fully connected		| input 120, output 80        									|
| RELU					|												|
| Dropout | keep probability 0.6 |
| Fully connected		| input 80, output 43        									|

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
