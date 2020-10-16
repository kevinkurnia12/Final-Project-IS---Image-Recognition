from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initialize the CNN
classifier = Sequential()

#Convolution2D
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

#Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#adding few layers
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#flatten
classifier.add(Flatten())

#full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#below is training data
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

#test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'Convolutional-Neural-Networks/Convolutional_Neural_Networks/dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

#below is for validation purposes, but it takes time to validate the training data which 
#may increase time up to triple.
#test_set = test_datagen.flow_from_directory(
#        'Convolutional-Neural-Networks/Convolutional_Neural_Networks/dataset/test_set',
#        target_size=(64, 64),
#        batch_size=32,
#        class_mode='binary')

classifier.fit_generator(
        training_set,
        samples_per_epoch=8000,
        nb_epoch=30)
        #originally below is for validation as well
        #validation_data = test_set,
        #nb_val_samples = 500)

import numpy as np
from keras.preprocessing import image
#below is the test using what the machine has learned above, we can try change 
#the filename as long it is in one folder with the code or properly directed
test = image.load_img('Cat03.jpg', target_size = (64, 64)) #cat pic
test = image.img_to_array(test)
test = np.expand_dims(test, axis = 0)
result = classifier.predict(test)
training_set.class_indices
if result[0][0] >= 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)

test = image.load_img('Mobile.jpg', target_size = (64, 64)) #dog pic
test = image.img_to_array(test)
test = np.expand_dims(test, axis = 0)
result = classifier.predict(test)
training_set.class_indices
if result[0][0] >= 0.5:
    prediction1 = 'dog'
else:
    prediction1 = 'cat'
print(prediction1)

test = image.load_img('cato.jpg', target_size = (64, 64))#cat pic
test = image.img_to_array(test)
test = np.expand_dims(test, axis = 0)
result = classifier.predict(test)
training_set.class_indices
if result[0][0] >= 0.5:
    prediction2 = 'dog'
else:
    prediction2 = 'cat'
print(prediction2)

test = image.load_img('random.png', target_size = (64, 64))#dog pic
test = image.img_to_array(test)
test = np.expand_dims(test, axis = 0)
result = classifier.predict(test)
training_set.class_indices
if result[0][0] >= 0.5:
    prediction3 = 'dog'
else:
    prediction3 = 'cat'
print(prediction3)

test = image.load_img('cattest.jpg', target_size = (64, 64))#cat pic
test = image.img_to_array(test)
test = np.expand_dims(test, axis = 0)
result = classifier.predict(test)
training_set.class_indices
if result[0][0] >= 0.5:
    prediction4 = 'dog'
else:
    prediction4 = 'cat'
print(prediction4)
