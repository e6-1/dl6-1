import numpy as np
import tensorflow as tf
import helpers as hp 


# Helper methods based on TensorFlow MNIST Convnet tutorial
def weight_variable(shape, mean=0.0, wd=None, name="weight"):
    initial = tf.truncated_normal(shape, mean=mean, stddev=0.1)
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)  # Store losses in a collection
    return tf.Variable(initial, name=name)


def bias_variable(shape, wd=None, name="bias"):
    initial = tf.constant(0.5, shape=shape)
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)  # Store losses in a collection
    return tf.Variable(initial, name=name)


def conv2d(x, W, name='conv'):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)


def max_pool_2x2(x, name='max_pool_2x2'):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def inference():

	image_input = tf.placeholder(tf.float32, [None, 28, 28, 1], name="images")
	gaze_input = tf.placeholder(tf.float32, [None, 2], name="gazes")
	brake_seq_input = tf.placeholder(tf.float32, [None, 3*2], name="brake_sequences")
	output = tf.placeholder(tf.float32, [None, 2], name="expected_braking")
	keep_prob = 0.5

	# Convolutional net for image processing

	# First layer
	W_conv1 = weight_variable([5, 5, 1, 32], name='image_W_conv1')  # 5x5x1 filter with 32 features
	b_conv1 = bias_variable([32], name='image_B_conv1')             # Bias for each filter

	h_conv1 = tf.nn.relu(conv2d(image_input, W_conv1) + b_conv1, name='conv1')
	h_pool1 = max_pool_2x2(h_conv1, name='pool1')

	# Second layer
	W_conv2 = weight_variable([5, 5, 32, 64], name='image_W_conv2')
	b_conv2 = bias_variable([64], name='image_B_conv2')

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name='conv2')
	h_pool2 = max_pool_2x2(h_conv2, name='pool2')

	# Fully-connected layer hidden layer
	W_fc1 = weight_variable([7 * 7 * 64, 1024], name='image_W_fc1')
	b_fc1 = bias_variable([1024], name='image_B_fc1')

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, 'hidden1')

	# Add dropout for fully-connected hidden layer
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	# Final logits
	W_fc2 = weight_variable([1024, 2], name='image_W_fc2')
	b_fc2 = bias_variable([2], name='image_B_fc2')

	image_logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


	# Logistic regression for human gaze
	W_g = weight_variable([2, 2], name='gaze_W_fc1')
	b_g = bias_variable([2], name='gaze_B_fc1')

	gaze_logits = tf.matmul(gaze_input, W_g) + b_g


	# Logistic regression for braking sequence
	W_bs = weight_variable([3*2, 2], name='bseq_W_fc1')
	b_bs = bias_variable([2], name='bseq_B_fc1')

	bs_logits = tf.matmul(brake_seq_input, W_bs) + b_bs

	# Weights for final logits
	image_logits_weights = weight_variable([2, 2], mean=0.3, name='image_logits_weight')
	gaze_logits_weights = weight_variable([2, 2], mean=0.3, name='gaze_logits_weight')
	bs_logits_weights = weight_variable([2, 2], mean=0.3, name='bs_logits_weights')
	bias = bias_variable([2], name='logits_bias')

	# Combine logistic and convnet
	logits = tf.add(tf.matmul(image_logits, image_logits_weights) + tf.matmul(gaze_logits, gaze_logits_weights) + tf.matmul(bs_logits, bs_logits_weights), bias)
	y = tf.nn.softmax(logits)
	return y


def loss(y):
	y_ = tf.get_default_graph().get_tensor_by_name('expected_braking:0')
	cross_entropy = tf.reduce_mean(tf.reduce_sum(-y_*tf.log(tf.clip_by_value(y, 1e-10,1.0)),reduction_indices=[1]))
	tf.add_to_collection('losses', cross_entropy)
	return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train(loss):
	train_op = tf.train.AdamOptimizer().minimize(loss)
	return train_op


def accuracy(y):
	y_ = tf.get_default_graph().get_tensor_by_name('expected_braking:0')
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return accuracy
