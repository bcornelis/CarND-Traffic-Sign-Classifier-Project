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
[german1]: ./external/sign1.png
[german2]: ./external/sign2.png
[german3]: ./external/sign3.png
[german4]: ./external/sign4.png
[german5]: ./external/sign5.png

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
-> n_train = len(X_train) = 34799
* The size of test set is:  <br>
-> n_test = len(X_test) = 12630
* The shape of a traffic sign image is: <br>
-> image_shape = X_train[0].shape = (32, 32, 3)
* The number of unique classes/labels in the data set is: <br>
-> n_classes = len(np.unique(y_train)) = 43

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

#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eleventh cell of the ipython notebook. 
The most important parameters are: 
* EPOCHS: 10
* BATCH_SIZE: 150
* learning rate: 0.005

To train the model, I used:
* for the optimizer: AdamOptimizer
* for the loss function: mean reduction of the cross entropy loss 

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the thirteenth cell of the Ipython notebook.

My final model results were:
* training set accuracy of: 0.931
* validation set accuracy of 0.931
* test set accuracy of: 0.914

If an iterative approach was chosen: <br>
* What was the first architecture that was tried and why was it chosen? <br>
-> The first architecture was the example LeNet from the course, without grayscale images, no normalization and no fake data
* What were some problems with the initial architecture? <br>
-> learning rate increase significantly in the beginning but always topped at around 80%
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.<br>
-> more data was generated<br>
-> from color space to grayscale<br>
-> from non-normalized to normalized images<br>
-> include dropout on different layers
* Which parameters were tuned? How were they adjusted and why? <br>
-> learning rate was increased from 0.001 to 0.005. Learning went faster, and the rate doesn't seem to high to result in a non-optimum<br>
-> epochs: too many epochs resulted in overfitting, to little in underfitting. 10 seemed an acceptable value to the results I get<br>
-> batch size: the higher the faster it learns, but needs more memory. 150 seems an acceptable value in combination with the nr of epochs

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?<br>
-> Convolutional layer is well suited as it searches for patterns (same weights and biases) anywere in the image, and that's in the end the goal of the CNN: find traffic signs anywhere in the image<br>
-> I've included two dropout layers (one for the first, and one for the second fully conntected layer) to make sure the network can also handle non-complete data.

If a well known architecture was chosen:
* What architecture was chosen?<br>
-> LeNet
* Why did you believe it would be relevant to the traffic sign application?<br>
-> The original goal of LeNet was to detect written letters. The goal is (about) the same: look anywhere in the image (convolution) for certain patterns
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?<br>
-> training,  validation and test accuracy are rougly the same. It's normal that the  test set accuracy is a little lower, as it's a data set which is completely new for the CNN.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![German Traffic Sign 1][german1] ![German Traffic Sign 2][german2] ![German Traffic Sign 3][german3] ![German Traffic Sign 4][german4] ![German Traffic Sign 5][german5]

There are some complexities which I expect with those images:
* resolution isn't always great
* there's some distortion on the signs

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the eightteenth cell of the Ipython notebook. The performance of the model was measured in the next cell.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| (9) No Passing      		| (32) End of all speed and passing limits   									| 
| (28) Children Crossing     			| (30) Beware of ice/snow										|
| (5) Speed Limit 80km/h					| (0) Speed limit 20km/h											|
| (17) No Entry	      		| (17) No Entry					 				|
| (33) Turn Right Ahead			| (33) Turn Right Ahead      							|

The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This does not really compare to the accuracy of the test/validation sets, which result in more than 90% accuracy.<br>

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 20th cell of the Ipython notebook.

The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .078         			| (32) End of all speed and passing limits    									| 
| .069     				| (9)	No passing									|
| .056					| (41)	End of no passing									|
| .037	      			| (19)	Dangerous curve to the left				 				|
| .023				    | (23) Slippery road      							|

For the second image ... 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .040         			| (30) Beware of ice/snow  									| 
| .036     				| (11) Right-of-way at the next intersection									|
| .029					| (12)	Priority road									|
| .011	      			| (21) Double curve					 				|
| .006				    | (40) Roundabout mandatory      							|

For the third image ... 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .075         			| (0) Speed limit (20km/h)  									| 
| .045     				| (1)	Speed limit (30km/h)									|
| .038					| (2)	Speed limit (50km/h)									|
| .090	      			| (24)	Road narrows on the right				 				|
| -.01				    | (26) Traffic signals     							|

For the fourth image ... 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .14         			| (17) No entry   									| 
| .066     				| (22)	Bumpy road									|
| .048					| (29)	Bicycles crossing									|
| .023	      			| (14)	Stop				 				|
| -.032				    | (13) Yield     							|

For the fifth image ... 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .029         			| (33) Turn right ahead  									| 
| .0096     				| (34) Ahead only										|
| .0051					| (37)	Go straight or left									|
| .0002	      			| (16)	Vehicles over 3.5 metric tons prohibited				 				|
| -.0004				    | (9) No passing     							|

To me it seems the algorithm is never really sure about it's predictions, and if the prediction is right, it's really not overwhelming...
