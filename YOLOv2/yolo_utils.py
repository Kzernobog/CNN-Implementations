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
        print(size)
        W = tf.get_variable(child.attrib['name'], size, initializer = tf.contrib.layers.xavier_initializer(seed = 0)) 
        parameters[child.attrib['name']] = W
        B = tf.Variable(tf.constant(0.01, shape=[size[-1]]))
        parameters['B'+(child.attrib['name'][1:])] = B
        
    return parameters