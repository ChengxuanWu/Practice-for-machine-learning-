#use keras api to build up a model
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(len(X_train), 28*28)/255
X_test = X_test.reshape(len(X_test), 28*28)/255
print('Rows: %d, Columns: %d' % (X_train.shape[0], X_train.shape[1]))
print('Rows: %d, Columns: %d' % (X_test.shape[0], X_test.shape[1]))

#turn the raw data into the normalized data
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)

X_train_centered = (X_train - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/std_val
del X_train, X_test

print(X_train_centered.shape, y_train.shape)
print(X_test_centered.shape, y_test.shape)
#setting the random seed to make sure the samples
np.random.seed(123)
tf.random.set_seed(123)
#use keras.utils.to_categorical to do one-hot labelizing
y_train_onehot = keras.utils.to_categorical(y_train)
print('The first 3 labels:', y_train_onehot[:3])
#build a model containing 3 layer of preception
#input layer - hidden layer - output layer
#the units would be 50-50-10, but the first layer have to accept all 784 features
#use keras's funtion: tanh: activation function / softmax: probabilies
model = keras.models.Sequential()
#add individual layer by moedl.add
model.add(
    keras.layers.Dense(
        units=50, input_dim= X_train_centered.shape[1],
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='tanh'))

model.add(
    keras.layers.Dense(
        units=50, input_dim=50, 
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='tanh'))

model.add(
    keras.layers.Dense(
        units= 10, input_dim=50,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='softmax'))

sgd_optimizer = keras.optimizers.SGD(
    learning_rate=0.001, decay=1e-7, momentum=0.9)
model.compile(optimizer= sgd_optimizer, loss='categorical_crossentropy')

history = model.fit(X_train_centered, y_train_onehot, batch_size=64,
                    epochs=50, verbose=1, validation_split=0.1)

y_train_pred = model.predict(X_train_centered, verbose=0)
#use the argmax to find the maximal value then trnasfer to the indeces
y_train_pred_class_indices = np.argmax(y_train_pred, axis=1)
print('The first 3 labels:', y_train_pred_class_indices[:3])

correct_preds = np.sum(y_train == y_train_pred_class_indices)
train_acc = correct_preds / y_train.shape[0]
print('Training accuracy: %.2f%%' % (train_acc * 100))

y_test_pred = model.predict(X_test_centered, verbose=0)
y_test_pred_class_indeces = np.argmax(y_test_pred, axis=1)
correct_preds = np.sum(y_test == y_test_pred_class_indeces)
test_acc = correct_preds/y_test.shape[0]
print('Test accuracy: %.2f%%' % (test_acc * 100))
























