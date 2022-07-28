import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Load data - split into train and test set
def load_data():
	(x_tr, y_tr), (x_test, y_test) = mnist.load_data()

	#print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
	#print('Test: X=%s, y=%s' % (testX.shape, testy.shape))

	# print out images
	for i in range(10):
		plt.subplot(330 + 1 + i)
		plt.imshow(x_tr[i], cmap=plt.get_cmap('gray'))
	plt.show()

	x_tr = x_tr.reshape(x_tr.shape[0], 28, 28, 1)
	x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
	input_shape = (28, 28, 1)

	y_tr = keras.utils.to.categorically(y_tr, 10)
	y_test = keras.utils.to.categorically(y_test, 10)

	return x_tr, x_test, y_tr, y_test

def data_prep():
	# Data preprocessing
	x_tr = x_tr.astype('float32')
	x_test = x_test.astype('float32')
	# normalize data
	x_tr /= 255
	x_test /= 255 
	# rotate
	# reflect 
	return x_tr, x_test

# Model creation
def define_model():
	batch_size = 128
	num_classes = 10
	epochs = 10

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	opt = SGD(learning_rate=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model 

# Evaluate model 
def evaluate_model(dataX, dataY, n_folds=5):
	scores, histories = list(), list()
	k_folds = KFold(n_folds, shuffle=True, random_state=1)
	for train_i, test_i in kfold.split(dataX):
		# define model
		model = define_model()
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_i], dataY[train_i], dataX[test_i], dataY[test_i]
		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
		# evaluate model
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		# store resulting scores
		scores.append(acc)
		histories.append(history)
	return scores, histories

# plot diagnostic learning curves
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		plt.subplot(2, 1, 1)
		plt.title('Cross Entropy Loss')
		plt.plot(histories[i].history['loss'], color='blue', label='train')
		plt.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		plt.subplot(2, 1, 2)
		plt.title('Classification Accuracy')
		plt.plot(histories[i].history['accuracy'], color='blue', label='train')
		plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	plt.show()

# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	plt.boxplot(scores)
	plt.show()
##plotting credits: https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/


# Main
#load data
trainX, trainY, testX, testY = load_data()
trainX, testX = data_prep()
scores, histories = evaluate_model()
summarize_diagnostics()
summarize_performance()

