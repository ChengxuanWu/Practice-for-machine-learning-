import tensorflow.compat.v1 as tf
import tensorflow as tf2
#the workflow of tensorflow
#step1: create a empty graph
g = tf.Graph()

with g.as_default():
    tf_a = tf.placeholder(tf.int32, shape=[], name='tf_a')
    tf_b = tf.placeholder(tf.int32, shape=[], name='tf_b')
    tf_c = tf.placeholder(tf.int32, shape=[], name='tf_c')
    
    r1 = tf_a - tf_b
    r2 = 2*r1
    z=r2+ tf_c
    
with tf.Session(graph=g) as sess:
    feed = {tf_a: 1,
            tf_b: 2,
            tf_c: 3}
    print('z:', sess.run(z, feed_dict=feed))
    
#try to use version to create the calculation
tf_a = tf2.constant(value=0, dtype=tf.int32, name='tf_a')
tf_b = tf2.constant(value=0, dtype=tf.int32, name='tf_b')
tf_c = tf2.constant(value=0, dtype=tf.int32, name='tf_c')
