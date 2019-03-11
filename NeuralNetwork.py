from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation
from keras.utils import to_categorical
from mlxtend.data import loadlocal_mnist

epochs = 50
batch_size = 10000
path = ""

# method for loading mnist data into numpy arrays
X_train, Y_train = loadlocal_mnist(images_path="train-images.idx3-ubyte",
                                   labels_path="train-labels.idx1-ubyte")
X_test, Y_test = loadlocal_mnist(images_path="test-images.idx3-ubyte",
                                 labels_path="test-labels.idx1-ubyte")

# turns labels to one-hot encodings
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

model = Sequential()

# input layer
model.add(Dense(784, input_shape=(784,)))
model.add(BatchNormalization())
model.add(Activation('relu'))

# hidden layer
model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation('relu'))

# output layer
model.add(Dense(10, activation="softmax"))

model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])

# start training
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1)

# save model's state for future use
model.save(path + "model.h5")

# evaluate model on test set
scores = model.evaluate(X_test, Y_test, verbose=1)

print("Test set scores", end='\n')
print("Loss: {0} - Accuracy: {1}%".format(scores[0], scores[1]*100))
