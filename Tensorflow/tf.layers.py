import tensorflow.compat.v1 as tf
import numpy as np
import tensorflow.keras as keras

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(len(X_train), 28*28)/255
X_test = X_test.reshape(len(X_test), 28*28)/255
print('Rows: %d, Columns: %d' % (X_train.shape[0], X_train.shape[1]))
print('Rows: %d, Columns: %d' % (X_test.shape[0], X_test.shape[1]))
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)
X_train_centered = (X_train - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/std_val
del X_train, X_test

print(X_train_centered.shape, y_train.shape)
print(X_test_centered.shape, y_test.shape)

#pixel 28*28 equals the number of features
n_features = X_train_centered.shape[1]
#finally the 10 outcome form 0 to 9 would be classified
n_classes =  10
#setting random seed
random_seed = 123
np.random.seed(random_seed)

g = tf.Graph()
with g.as_default():
    tf.set_random_seed(random_seed)
    tf_x = tf.placeholder(dtype= tf.float32, shape=(None, n_features),
                          name='tf_x')
    tf_y = tf.placeholder(dtype= tf.int32, shape=(None),
                          name='tf_y')
    
    y_onehot = tf.one_hot(indices= tf_y, depth= n_classes)
    
    #layers on version1 was deprecated, so use keras as alternative way
    #use input behind the layers
    h1 = keras.layers.Dense(units=50, activation="tanh", name="layer1")(tf_x)
    
    h2 = keras.layers.Dense(units=50, activation= tf.tanh, name= 'layer2')(h1)
    
    logits = keras.layers.Dense(units=10, activation=None, name= 'layer3')(h2)
    
    predictions = {'classes': tf.argmax(logits, axis=1, name='predicted_classes'),
                   'probabilities':tf.nn.softmax(logits, name='softmax_tensor')}
    
#define cost function and optimizer
with g.as_default():
    cost = tf.losses.softmax_cross_entropy(onehot_labels=y_onehot, logits=logits)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
    train_op = optimizer.minimize(loss= cost)
    init_op = tf.global_variables_initializer()

def create_batch_generator(X, y, batch_size=128, shuffle=False):
    X_copy = np.array(X)
    y_copy = np.array(y)
    
    if shuffle:
        data = np.column_stack((X_copy, y_copy))
        np.random.shuffle(data)
        X_copy= data[:, :-1]
        y_copy= data[:, -1].astype(int)
    
    for i in range(0, X.shape[0], batch_size):
        #yield function could produce a sequence as a generator
        #could seen as a iterator, also could use next to print data respectively
        yield(X_copy[i:i+batch_size], y_copy[i:i+batch_size])

sess = tf.Session(graph=g)
sess.run(init_op)

for epoch in range(50):
    training_costs = []
    batch_generator = create_batch_generator(X_train_centered, y_train, batch_size=64)
    for batch_X, batch_y in batch_generator:
        feed = {tf_x:batch_X, tf_y:batch_y}
        _, batch_cost = sess.run([train_op, cost], feed_dict=feed)
        training_costs.append(batch_cost)
    print('--Epoch %2d ''Avg. Training Loss: %.4f' % (epoch+1, np.mean(training_costs)))
    
#fulfill the prediction on test set:
feed = {tf_x: X_test_centered}
y_pred = sess.run(predictions['classes'], feed_dict=feed)
print('Test Accuracy: %.2f%%' % (100*np.sum(y_pred == y_test)/y_test.shape[0]))

        

    
    
    
    
    