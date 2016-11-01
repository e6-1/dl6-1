import numpy as np
import pandas as pd


def get_sub_seq(seq, start, end):
    """Get the sub sequence starting at the start index and ending at the end index."""
    arr = seq[max([0, start]):end]
    if start < 0:
        arr = np.append(np.zeros((abs(start),2)), arr, axis=0)
    for i in range(len(arr)):
        if np.sum(arr[i]) == 0:
            arr[i] = [1, 0]
    return arr


def get_train_test_split(train_size, test_size):
	inds = range(train_size + test_size)
	test_inds = minibatch(inds, test_size, train_size + test_size)[0]
	training_inds = [i for i in inds if i not in test_inds]
	return training_inds, test_inds


def minibatch(data, batch_size, data_size):
    """Generates a minibatch from the given data and parameters."""
    randomized = np.random.permutation(data)
    batches = []
    num_batches = 0
    while num_batches * batch_size < data_size:
        new_batch = randomized[num_batches * batch_size:(num_batches + 1) * batch_size]
        batches.append(new_batch)
        num_batches += 1
    return batches


def get_glimpses(images, coords):
    """Gets a batch of glimpses."""
    arr = []
    for img, coord in zip(images, coords):
        arr.append(get_glimpse(img, coord[0], coord[1]))
    return np.array(arr)


def get_glimpse(image, x, y, stride=14):
    """Returns a subsection (glimpse) of the image centered on the given point."""
    x = int(x)  # Force to int
    y = int(y)  # Force to int
    min_x = x - stride
    max_x = x + stride
    
    min_y = y - stride
    max_y = y + stride
    image_glimpse = image[min_y:max_y, min_x:max_x, :]  # NOTE: row, column, RGB
#     image_glimpse = image[min_y:max_y, min_x:max_x, 0]  # NOTE: row, column, RGB; everything is greyscale; flatten RGB layer
    return imgToArr(image_glimpse)


def get_data():
	"""Returns a dictionary of data with keys for "inputs" and "outputs"."""
	input_glimpses = np.zeros((80000, 28, 28, 3))
	input_gazes = np.zeros((80000, 2))
	outputs = np.zeros((80000, 2))
	for batch in range(1, 9):
	    file_name = "data/glimpse_batchc_{0}.npz".format(batch)
	    array = np.load(file_name)
	    input_glimpses[(batch - 1) * 10000: batch * 10000] = array['frames']
	    input_gazes[(batch - 1) * 10000: batch * 10000] = array['gazes']
	    outputs[(batch - 1) * 10000: batch * 10000] = array['braking']

	for i in range(len(outputs)):
	    if np.sum(outputs[i]) == 0:
	        outputs[i] = [1, 0]

	sequences = np.array([get_sub_seq(outputs, i-3, i) for i in range(len(outputs))])
	sequences = sequences.reshape(-1, 3*2)

	data = {
		"input_glimpses": input_glimpses,
		"input_gazes": input_gazes,
		"input_sequences": sequences,
		"outputs": outputs
	}
	return data
