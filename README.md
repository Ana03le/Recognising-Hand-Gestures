# Recognising-Hand-Gestures
Project: Recognizing Hand Gestures
Introduction:
In the current world of automation, voice recognition is used to perform tasks like playing music, lowering room temperature etc. (example Alexa). This idea can further be extended to use hand gestures for performing tasks like switching off of light, closing door, applying lipstick, applying eye make etc. Keeping these uses in mind, we have classified hand gesture inputs which is further used for classifying and predicting actions like Baby Crawling, Balancing Beam and Archery.
Input:
We have implemented neural network which performs hand gestures recognition for data from UCF data set. This data set is well competitive because of 13,320 videos from 101 action categories. It
also provides the largest diversity in terms of actions and with the presence of large variations in camera motion, object appearance and pose, object scale, viewpoint, cluttered background. http://crcv.ucf.edu/data/UCF101.php
Architecture:
We have developed a neural network model with a multiple, alternate layers of 2D-Convolution and 2D MaxPooling. We have used “relu” as an activation function. Next layers are added sequentially for flattening the data, applying LSTM and then final layer of Softmax.
  Flow Structure:
1.Data segregation into test, train
2.Extraction of images from Videos (using ffmpeg) 3.Data Cleaning (encode one hot, rescaling) 4.Extraction of frames from images
5.Training the model
6.Predicting
Libraries used:
tqdm
keras.preprocessing (Image Generator) keras.applications.inception_v3
keras.models
keras.layers
keras.optimizers
keras.layers.wrappers
keras.layers.convolutional (MaxPooling2D, Conv2D)

 Actual Neural Network Design Code:
def lrcn(self):
model = Sequential()
model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), activation='relu', padding='same'), input_shape=self.input_shape))
model.add(TimeDistributed(Conv2D(32, (3,3), kernel_initializer="he_normal", activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))) model.add(TimeDistributed(Conv2D(64, (3,3),
 padding='same', activation='relu'))) model.add(TimeDistributed(Conv2D(64, (3,3),
padding='same', activation='relu'))) model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
model.add(TimeDistributed(Conv2D(128, (3,3), padding='same', activation='relu')))
model.add(TimeDistributed(Conv2D(128, (3,3), padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
model.add(TimeDistributed(Conv2D(256, (3,3), padding='same', activation='relu')))
model.add(TimeDistributed(Conv2D(256, (3,3), padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
model.add(TimeDistributed(Conv2D(512, (3,3), padding='same', activation='relu')))
model.add(TimeDistributed(Conv2D(512, (3,3), padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))) model.add(TimeDistributed(Flatten()))
 model.add(Dropout(0.5))
model.add(LSTM(256, return_sequences=False, dropout=0.5)) model.add(Dense(self.nb_classes, activation='softmax'))


 Evaluation and Prediction:
The system will predict values of hand gestures and classify the input accordingly. The accuracy of the system will be measured based on correctly predicting the input gesture.
Further Changes and Experiment:
We are further experimenting to gain results of model prediction by applying different network architecture using 3D Convolutional and Max Pooling.


References:
https://blog.coast.ai/five-video-classification-methods-implemented-in-keras-and-tensorflow- 99cad29cc0b5
https://medium.com/twentybn/gesture-recognition-using-end-to-end-learning-from-a-large-video- database-2ecbfb4659ff
   
https://www.cv- foundation.org/openaccess/content_cvpr_workshops_2015/W15/papers/Molchanov_Hand_Gesture_Reco gnition_2015_CVPR_paper.pdf
https://www.sciencedirect.com/science/article/abs/pii/S0952197609000694
