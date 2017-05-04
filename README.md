# Traffic Sign Recognition 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[hist]: ./images/hist.png "Distribution of classes in train, test, valid"
[goodex]: ./images/good.png "Good predictions"
[badex]:  ./images/bad.png "Good predictions"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


### Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of testing examples = 12630
* Number of validation examples = 4410
* Image data shape = (32, 32, 3)
* Number of classes = 43


#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram showing the distribution of the different classes in the train, test and validation datasets. The distribution seems similar across the 3 datasets.

![Data distribution][hist]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? 
I applied per-image per-channel normalization to make the input 0-mean and unit standard deviation. I computed for each image, mean and std for every channel, and then normalized as `z = (x - mean) / sd`. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model was simply a LeNet model with dropout regularization at the second last layer.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1  stride, outputs 10x10x16      									|
| Max pooling               | 2x2 stride, outputs 5x5x16
| Fully connected		|  Of size 400 x 120      									|
| RELU                                  |                                  |
| Fully connected                |  120x84     |
| RELU                     |   |
| dropout                  | dropout regularization was important for accurary         |
| Fully connected               | 84 X 43    |
| Softmax				|           									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a CPU. Set the following parameters
* learning rate 0.001 
* Batch size = 128 
* 10 epochs

It rook on average 7-8 minutes for training 10 epochs.

Holdout regularization played a big role in improving validation accuracy.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.945 
* test set accuracy of 0.937

I started with LeNet model  since it is a well known model for this kind of problem. Initial validation accuracy was around 0.90-0.91. To get the validation accuracy up, I added dropout in the second last layer. This immediately improved results to above 0.94. 

Dropout seemed like a strong regularization technique, so tried that first. And it worked. 

The preprocessing that I did also may have played a role in making this easier.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web: Please see the accompanying iPy notebook for all the images.

Some images are unseen (the circular image with pedestrian) so don't expect that to be caught by the model.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (50km/h)      		| Speed limit (80km/h)   									| 
| Road work     			| Road work 										|
| Pedestrians					| Roundabout mandatory											|
| Pedestrians	      		| General caution					 				|
|  No passing for vehicles over 3.5 metric tons			|  No passing for vehicles over 3.5 metric tons      							|
| Bicycles crossing | Bicycles crossing   |
| Ahead only | Ahead only |
| Turn left ahead | Turn left ahead | 


The model was able to correctly guess 5 of the 8 traffic signs, which gives an accuracy of 62.5%. The images in this dataset are harder because many of them are unseen. Also, many of them don't have representatives in the training data.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 578th and 611th cell of the Ipython notebook.

Some good examples.
![Good examples][badex]

Some bad examples.
![Bad examples][goodex]




