# **Behavioral Cloning** 

## Writeup Report
### udacity project 3 homework
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

According to this paper:End to End Learning for Self-Driving Cars(NVIDIA Corporation)
My model consists of five convolution neural networks:
1. 24x5x5,strides(2x2).activation(relu)
2. 36x5x5,strides(2x2),activation(relu)
3. 48x5x5, strides(2x2),activation(relu)
4. 64x3x3, activation(relu)
5. 64x3x3, activation(relu) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 39). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


#### 3. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Train several laps to collect enough data.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was according to input data and output data.Because input data is image and output is to control vehcile, control vehcile depends on current image and output directly.So we choose CNN to solve this problem.

My first step was to use a convolution neural network model similar to the Lenet.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a high mean squared error on the training set and the validation set. This implied that the model was not powerful enough. 

Then I change the CNN to use a convolution neural network model similar to the Paper that writen by Nvidia,I thought this model might be appropriate because this has been tested in self driving car before.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track because there is lack of data at the corner, to improve the driving behavior in these cases, I collect more data from the corner.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 38-50) consisted of a convolution neural network with the following layers one normalization 5 CNN layers and one Flatten ,three Fully-connected layters.

Here is a visualization of the architecture

![CNN layers](https://github.com/rzhengyang/CarND-Behavioral-Cloning-P3-master/blob/master/img/CNN%20layers.jpg)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![center][https://github.com/rzhengyang/CarND-Behavioral-Cloning-P3-master/blob/master/img/center.jpg]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to turn from the side recovery to the center.
 These images show what a recovery looks like starting from the road side:


![1](https://github.com/rzhengyang/CarND-Behavioral-Cloning-P3-master/blob/master/img/1/1.jpg)
![2](https://github.com/rzhengyang/CarND-Behavioral-Cloning-P3-master/blob/master/img/1/2.jpg)
![3](https://github.com/rzhengyang/CarND-Behavioral-Cloning-P3-master/blob/master/img/1/3.jpg)
![4](https://github.com/rzhengyang/CarND-Behavioral-Cloning-P3-master/blob/master/img/1/4.jpg)
![5](https://github.com/rzhengyang/CarND-Behavioral-Cloning-P3-master/blob/master/img/1/5.jpg)
![6](https://github.com/rzhengyang/CarND-Behavioral-Cloning-P3-master/blob/master/img/1/6.jpg)

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would helpful to solve  emergency problems. For example, here is an image that has then been flipped:
![1](https://github.com/rzhengyang/CarND-Behavioral-Cloning-P3-master/blob/master/img/2/1.jpg)
![2](https://github.com/rzhengyang/CarND-Behavioral-Cloning-P3-master/blob/master/img/2/2.jpg)
![3](https://github.com/rzhengyang/CarND-Behavioral-Cloning-P3-master/blob/master/img/2/3.jpg)
![4](https://github.com/rzhengyang/CarND-Behavioral-Cloning-P3-master/blob/master/img/2/4.jpg)
![5](https://github.com/rzhengyang/CarND-Behavioral-Cloning-P3-master/blob/master/img/2/5.jpg)
![6](https://github.com/rzhengyang/CarND-Behavioral-Cloning-P3-master/blob/master/img/2/6.jpg)

And then I use three cameras images to train CNN.

After the collection process, I had 7668 number of data points. I then preprocessed this data by cut the top 75 about sky and bottom 25 to keep road image. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3. I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### 4. Validation Loss Metrics
![validation](https://github.com/rzhengyang/CarND-Behavioral-Cloning-P3-master/blob/master/img/1.png)