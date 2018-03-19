import xml.etree.ElementTree as ET
import tensorflow as tf
import numpy as np

def initialize_weights(xml_file):
    '''
    Reads model parameter weights from xml_file and initializes filters and biases
    
    Args:
    xml_file - configuration xml with absolute path
    
    Returns:
    parameters - a dictionary containing initialized parameters
    '''
    parameters = {}
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for child in root:
        size = []
        for child1 in child:
            # print(child.attrib['name'], child1.tag, child1.text)
            if (child1.tag == 'dimension'):
                size.append((int)(child1.text))
                size.append((int)(child1.text))
            if (child1.tag == 'input'):
                size.append((int)(child1.text))
            if (child1.tag == 'output'):
                size.append((int)(child1.text))
        
        W = tf.get_variable(child.attrib['name'], size, initializer = tf.contrib.layers.xavier_initializer(seed = 0)) 
        parameters[child.attrib['name']] = W
        B = tf.get_variable('b'+(child.attrib['name'][1:]), [size[-1],1], initializer = tf.zeros_initializer())
        parameters['b'+(child.attrib['name'][1:])] = B
        print(size, child.attrib['name'], 'b'+(child.attrib['name'][1:]))
        
    return parameters



def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitioning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def conv_layer(A_p, W, B, strides=[1,1,1,1], padding="SAME", name="default", activation='relu'):
    '''
    A_p = activation of the previous layer
    W = Filter to convolve
    B = bias term
    acitvation = type of activation
    '''
    #tf.name_scope creates namespace for operators in the default graph, places into group, easier to read
    with tf.name_scope('conv_'+name):
        conv = tf.nn.conv2d(A_p, W, strides=strides, padding=padding)

        if (activation == 'relu'):
            act = tf.nn.relu(tf.nn.bias_add(conv, B))
        elif (activation == 'leaky_relu'):
            act = tf.nn.leaky_relu(tf.nn.bias_add(conv, B))


        #visualize the the distribution of weights, biases and activations
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", B)
        tf.summary.histogram("activations", act)
        #return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    return act
 


def fc_layer(A_p, output_num, activation_fn=None, name="default"):
    """
    A_p = activations of the previous layer
    output_num = number of neurons in the fully connected layer
    """
    with tf.name_scope("fc_"+name):
        #fully connected part
        FC1 = tf.contrib.layers.fully_connected(A_p, ouput_num, activation_fn=activation_fn)
        
    return FC1

def max_pool(A_p, kernel, strides, padding="SAME", name="default"):
    """
    A_p = activation of the previous layer
    kernel = size of the filter
    strides = strides of the pooling filter
    """
    with tf.name_scope("pool_"+name):
        P = tf.nn.max_pool(A_P, kernel, strides, padding=padding)
    return P