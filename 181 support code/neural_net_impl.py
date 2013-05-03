from neural_net import *
import random
import numpy # added to do arange
import math


def FeedForward(network, input, network_type):
  """
  Arguments:
  ---------
  network : a NeuralNetwork instance
  input   : an Input instance
  network_type  : a string indicating network type

  Returns:
  --------
  Nothing

  Description:
  -----------
  This function propagates the inputs through the network. That is,
  it modifies the *raw_value* and *transformed_value* attributes of the
  nodes in the network, starting from the input nodes.

  Notes:
  -----
  The *input* arguments is an instance of Input, and contains just one
  attribute, *values*, which is a list of pixel values. The list is the
  same length as the number of input nodes in the network.

  i.e: len(input.values) == len(network.inputs)

  This is a distributed input encoding (see lecture notes 7 for more
  informations on encoding)

  In particular, you should initialize the input nodes using these input
  values:

  network.inputs[i].raw_value = input[i]
  """
  network.CheckComplete()

  # 1) Assign input values to input nodes  

  input_length = len(network.inputs) 
  for i in range(input_length):
    network.inputs[i].raw_value = input[i]
    network.inputs[i].transformed_value = input[i]

  if network_type == 'custom':
    # 1a) Propagate to layer 1
    layer1_length = len(network.layer1_nodes)
    for l1 in range(layer1_length):
      network.layer1_nodes[l1].raw_value = \
      network.ComputeRawValue(network.layer1_nodes[l1])    
      network.layer1_nodes[l1].transformed_value = \
      network.Sigmoid(network.layer1_nodes[l1].raw_value)

    # 1b) Propagate to layer 2
    layer2_length = len(network.layer2_nodes)
    for l2 in range(layer2_length):
      network.layer2_nodes[l2].raw_value = \
      network.ComputeRawValue(network.layer2_nodes[l2])    
      network.layer2_nodes[l2].transformed_value = \
      network.Sigmoid(network.layer2_nodes[l2].raw_value)

  # 2) Propagates to hidden layer
  hidden_length = len(network.hidden_nodes)
  for j in range(hidden_length):
    network.hidden_nodes[j].raw_value = \
    network.ComputeRawValue(network.hidden_nodes[j])
    network.hidden_nodes[j].transformed_value = \
    network.Sigmoid(network.hidden_nodes[j].raw_value)

  # 3) Propagates to the output layer
  output_length = len(network.outputs)
  for k in range(output_length):
    network.outputs[k].raw_value = network.ComputeRawValue(network.outputs[k])
    network.outputs[k].transformed_value = \
    network.Sigmoid(network.outputs[k].raw_value)



def Backprop(network, input, target, learning_rate, network_type):
  """
  Arguments:
  ---------
  network       : a NeuralNetwork instance
  input         : an Input instance
  target        : a target instance
  learning_rate : the learning rate (a float)
  network_type  : a string indicating network type

  Returns:
  -------
  Nothing

  Description:
  -----------
  The function first propagates the inputs through the network
  using the Feedforward function, then backtracks and update the
  weights.

  Notes:
  ------
  The remarks made for *FeedForward* hold here too.

  The *target* argument is an instance of the class *Target* and
  has one attribute, *values*, which has the same length as the
  number of output nodes in the network.

  i.e: len(target.values) == len(network.outputs)

  In the distributed output encoding scenario, the target.values
  list has 10 elements.

  When computing the error of the output node, you should consider
  that for each output node, the target (that is, the true output)
  is target[i], and the predicted output is network.outputs[i].transformed_value.
  In particular, the error should be a function of:

  target[i] - network.outputs[i].transformed_value
  
  """
  network.CheckComplete()
  # 1) We first propagate the input through the network
  # 2) Then we compute the errors and update the weigths starting with the last layer
  # 3) We now propagate the errors to the hidden layer, and update the weights there too
  
  FeedForward(network, input, network_type)

  ### Calculate errors ###

  # calculate error at output
  error_output = []
  output_length = len(network.outputs)
  for k in range(output_length):
    error_output.append((target[k] - network.outputs[k].transformed_value) * \
    network.outputs[k].transformed_value * (1 - network.outputs[k].transformed_value))

  # calculate error in hidden layer using error in output layer
  error_hidden = []

  sum = [0] * len(network.hidden_nodes)
  hidden_length = len(network.hidden_nodes)
  for j in range(hidden_length):
    for k in range(output_length):
      sum[j] += network.outputs[k].weights[j].value * error_output[k]

  for j in range(hidden_length):
    error_hidden.append(network.hidden_nodes[j].transformed_value * \
    (1 - network.hidden_nodes[j].transformed_value) * sum[j])


  if network_type == 'custom':
    # calculate error in layer 2 using error in hidden layer
    layer2_length = len(network.layer2_nodes)
    error_l2 = []
    sum = [0] * layer2_length
    
    for l2 in range(layer2_length):
      for j in range(hidden_length):
        sum[l2] += network.hidden_nodes[j].weights[l2].value * error_hidden[j]

    for l2 in range(layer2_length):
      error_l2.append(network.layer2_nodes[l2].transformed_value * \
      (1 - network.layer2_nodes[l2].transformed_value) * sum[l2])

    # calculate error in layer 1 using error in layer 2
    layer1_length = len(network.layer1_nodes)
    error_l1 = []
    sum = [0] * layer1_length

    for l1 in range(layer1_length):
      for l2 in range(layer2_length):
        if not network.layer2_nodes[l2].weights:
          sum[l1] += network.layer2_nodes[l2].weights[l1].value * error_l2[l2]

    for l1 in range(layer1_length):
      error_l1.append(network.layer1_nodes[l1].transformed_value * \
      (1 - network.layer1_nodes[l1].transformed_value) * sum[l1])

  ### Update weights ###

  # update output layer weights
  input_length = len(network.inputs)

  for k in range(output_length):
    for j in range(hidden_length):
      network.outputs[k].weights[j].value += learning_rate * error_output[k] * \
      network.hidden_nodes[j].transformed_value

  if network_type == 'custom':
    # update hidden layer weights with previous layer as layer2
    for j in range(hidden_length):
      for l2 in range(layer2_length):
        network.hidden_nodes[j].weights[l2].value += \
        learning_rate * error_hidden[j] * network.layer2_nodes[l2].transformed_value

    # update layer2 weights with previous layer as layer 1
    for l2 in range(layer2_length):
      for l1 in range(layer1_length):
        network.layer2_nodes[l2].weights[l1].value += \
        learning_rate * error_l2[l2] * network.layer1_nodes[l1].transformed_value

    # update layer1 weights with previous layer as inputs
    for l1 in range(layer1_length):
      for i in range(input_length):
        if not network.layer1_nodes[l1].weights:
          network.layer1_nodes[l1].weights[i].value += \
          learning_rate * error_l1[l1] * network.inputs[i].transformed_value

  else:
    # update hidden layer weights with previous layer as inputs
    for j in range(hidden_length):
      for i in range(input_length):
        network.hidden_nodes[j].weights[i].value += \
        learning_rate * error_hidden[j] * network.inputs[i].transformed_value


def Train(network, inputs, targets, learning_rate, epochs, network_type):
  """
  Arguments:
  ---------
  network       : a NeuralNetwork instance
  inputs        : a list of Input instances
  targets       : a list of Target instances
  learning_rate : a learning_rate (a float)
  epochs        : a number of epochs (an integer)
  network_type  : a string indicating network type

  Returns:
  -------
  Nothing

  Description:
  -----------
  This function should train the network for a given number of epochs. That is,
  run the *Backprop* over the training set *epochs*-times
  """

  #for e in range(epochs):
  input_length = len(inputs)
  for i in range(input_length):
    Backprop(network, inputs[i], targets[i], learning_rate, network_type)

  network.CheckComplete()
  pass
  


class EncodedNetworkFramework(NetworkFramework):
  def __init__(self):
    """
    Initialization.
    YOU DO NOT NEED TO MODIFY THIS __init__ method
    """
    super(EncodedNetworkFramework, self).__init__() 

  def EncodeLabel(self, label):
    
    """
    Arguments:
    ---------
    label: a number between 0 and 9

    Returns:
    ---------
    a list of length 10 representing the distributed
    encoding of the output.

    Description:
    -----------
    Computes the distributed encoding of a given label.

    Example:
    -------
    0 => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    3 => [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    Notes:
    ----
    Make sure that the elements of the encoding are floats.
    
    """
    encodedlabel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    encodedlabel[label] = 1.0
    return encodedlabel

  def GetNetworkLabel(self):
    """
    Arguments:
    ---------
    Nothing

    Returns:
    -------
    the 'best matching' label corresponding to the current output encoding

    Description:
    -----------
    The function looks for the transformed_value of each output, then decides 
    which label to attribute to this list of outputs. The idea is to 'line up'
    the outputs, and consider that the label is the index of the output with the
    highest *transformed_value* attribute

    Example:
    -------

    # Imagine that we have:
    map(lambda node: node.transformed_value, self.network.outputs) => 
    [0.2, 0.1, 0.01, 0.7, 0.23, 0.31, 0, 0, 0, 0.1, 0]

    # Then the returned value (i.e, the label) should be the index of the item 0.7,
    # which is 3
    
    """
    outputlist = map(lambda node: node.transformed_value, self.network.outputs)
    return outputlist.index(max(outputlist))


  def Convert(self, image):
    """
    Arguments:
    ---------
    image: an Image instance

    Returns:
    -------
    an instance of Input

    Description:
    -----------
    The *image* arguments has 2 attributes: *label* which indicates
    the digit represented by the image, and *pixels* a matrix 14 x 14
    represented by a list (first list is the first row, second list the
    second row, ... ), containing numbers whose values are comprised
    between 0 and 256.0. The function transforms this into a unique list
    of 14 x 14 items, with normalized values (that is, the maximum possible
    value should be 1).
    
    """

    image_width = len(image.pixels)
    outputlist = [0.] * (image_width+1)

    for i in range(image_width):
      outputlist.append(0.)
      for j in range(image_width):
        outputlist.append(image.pixels[i][j]/256)

    return outputlist

  def InitializeWeights(self):
    """
    Arguments:
    ---------
    Nothing

    Returns:
    -------
    Nothing

    Description:
    -----------
    Initializes the weights with random values between [-0.01, 0.01].

    Hint:
    -----
    Consider the *random* module. You may use the the *weights* attribute
    of self.network.
    
    """

    for weight in self.network.weights:
      weight.value = random.uniform(-0.01,0.01)



class SimpleNetwork(EncodedNetworkFramework):
  def __init__(self):
    """
    Arguments:
    ---------
    Nothing

    Returns:
    -------
    Nothing

    Description:
    -----------
    Initializes a simple network, with 196 input nodes,
    10 output nodes, and NO hidden nodes. Each input node
    should be connected to every output node.
    """
    super(SimpleNetwork, self).__init__() 
    
    pass



class HiddenNetwork(EncodedNetworkFramework):
  def __init__(self, number_of_hidden_nodes=15):
    """
    Arguments:
    ---------
    number_of_hidden_nodes : the number of hidden nodes to create (an integer)

    Returns:
    -------
    Nothing

    Description:
    -----------
    Initializes a network with a hidden layer. The network
    should have 196 input nodes, the specified number of
    hidden nodes, and 10 output nodes. The network should be,
    again, fully connected. That is, each input node is connected
    to every hidden node, and each hidden_node is connected to
    every output node.
    """
    super(HiddenNetwork, self).__init__() # < Don't remove this line >

    # 1) Adds an input node for each pixel
    for i in range(225):
      self.network.AddNode(Node(), 1) #use NeuralNetwork?

    # 2) Adds the hidden layer
    for j in range(number_of_hidden_nodes):
      self.network.AddNode(Node(), 4)

    # 3) Adds an output node for each possible digit label.
    for k in range(10):
      self.network.AddNode(Node(), 5)
    
    # connect nodes in hidden layer to input layer
    for j in range(number_of_hidden_nodes):
      for i in range(225): 
        self.network.hidden_nodes[j].AddInput(self.network.inputs[i], 0, self.network)

    # connect nodes in output layer to hidden layer
    for k in range(10):
      for j in range(number_of_hidden_nodes): 
        self.network.outputs[k].AddInput(self.network.hidden_nodes[j], 0, self.network)      

    pass
    


class CustomNetwork(EncodedNetworkFramework):
  def __init__(self, kernel_width=3, kernel_offset=2, 
              number_of_fm_l1 = 6, number_of_fm_l2 = 50,
              number_of_hidden_nodes = 100):
    """
    Arguments:
    ---------
    Your pick.

    Returns:
    --------
    Your pick

    Description:
    -----------
    Surprise me!
    """
    super(CustomNetwork, self).__init__() 

    ### Create proper size for shared weight globals ###
    number_of_nodes_kernel = kernel_width * kernel_width

    global shared_weights_1 
    shared_weights_1 = [Weight(0.0)] * (number_of_nodes_kernel * number_of_fm_l1)
    global shared_weights_2 
    shared_weights_2 = [Weight(0.0)] * (number_of_nodes_kernel * number_of_fm_l2)

    ### Add appropriate number of nodes for each layer ###

    # 1) Adds an input node for each pixel
    for i in range(225): # using 15x15
      self.network.AddNode(Node(), 1)

    # 2) Adds layer 1 (1st convolution)
    layer1_width = int(math.ceil((15 - kernel_width + 1.) / kernel_offset))
    assert layer1_width == 7, '1st layer must be 7x7'
    layer1_size = layer1_width * layer1_width
    number_of_layer1_nodes = layer1_size * number_of_fm_l1

    for l1 in range(number_of_layer1_nodes):
      self.network.AddNode(Node(), 2) 

    # 2) Adds layer 2 (2nd convolution)
    layer2_width = int(math.ceil((layer1_width - kernel_width + 1.) / kernel_offset))
    assert layer2_width == 3, '2nd layer must be 3x3'
    layer2_size = layer2_width * layer2_width
    number_of_layer2_nodes = layer2_size * number_of_fm_l2

    for l2 in range(number_of_layer2_nodes):
      self.network.AddNode(Node(), 3) 

    # 3) Add the hidden layer
    for j in range(number_of_hidden_nodes):
      self.network.AddNode(Node(), 4)

    # 4) Adds an output node for each possible digit label.
    for k in range(10):
      self.network.AddNode(Node(), 5)
  
    

    
    ### Connect nodes in layer with previous layer ###

    #have we properly added convolutional layer 0 to 1 below? Did I accidentally mess up any code? Please check conflicts!!!
    #WE CAN MAKE THIS FUNCTIONAL it would be great.
    #define function
    #def add_layers(fmsize1,number_of_layer2_nodes,number_of_fm_l1,kernal_width,kernal_offset):
    #call function
    #add_layers(49,number_of_layer2_nodes,number_of_fm_l1,3,2)

    
    # connect nodes in layer1 to input layer
    
    current_node_1 = 0

    saved_kernels_1 = []

    for m in range(number_of_fm_l1):
      current = 0
      begin = 0

      for j in range(layer1_width):

        for node in numpy.arange(current_node_1, (current_node_1 + layer1_width)):          
            # each horizontal
          for across in range(kernel_width):
              
              # each cell in row
            for pixel in numpy.arange(current, (current + kernel_width)):
              #print "pixel:", pixel              
              saved_kernels_1.append(self.network.layer1_nodes[int(pixel)])
              
            current += 30
  
          # print "node:", node
          print len(saved_kernels_1)
          self.network.layer1_nodes[node].AddConvolutionalInput(saved_kernels_1, shared_weights_1, self.network)
          begin += kernel_offset
          current = begin
          saved_kernels_1 = []

        current_node_1 += layer1_width
        #print "current node:",  current_node_1

        begin += 16
        #print "begin", begin
        current = begin




    # connect nodes in layer 2 to layer 1 

    current_node_2 = 0

    saved_kernels_2 = []

    for m in range(number_of_fm_l2):
      current = 0
      begin = 0

      for j in range(layer2_width):

        for node in numpy.arange(current_node_2, (current_node_2 + 3)):

          # each feature map in previous layer
          for m in range(number_of_fm_l1):
          
            # each horizontal
            for across in range(kernel_width):
              
              # each cell in row
              for pixel in numpy.arange(current, (current + kernel_width)):            
                saved_kernels_2.append(self.network.layer1_nodes[int(pixel)])
              
              current += 7

            current += 28

          #print "node:", node
          self.network.layer2_nodes[node].AddConvolutionalInput(saved_kernels_2, shared_weights_2, self.network)
          begin += kernel_offset
          current = begin
          saved_kernels_2 = []
 
        current_node_2 += layer2_width
        #print "current node:",  current_node_2
        begin += 8
        #print "begin", begin
        current = begin


    

    """
    # connect nodes in layer 2 to layer 1 
    #hard coded - does this variable already exist?
    fmsize1 = 49

    rtfm1 = math.sqrt(fmsize1)


    #marks node position in layer 1
    current = 0
    #marks topleft cell of current kernel
    topleft = 0

    #array of nodes from each layer 1 fm to store in layer 2 node
    nodes_for_fm = []
    #loop through layer 2 nodes
    for node in range(number_of_layer2_nodes):

      #loop through every feature map in layer 1
      for fm in range(number_of_fm_l1):

        #each vertical displacement of kernel
        for down in range(kernel_width):

          #each horizontal
          for across in range(kernel_width):

            #each each row in kernel
            for down in range(kernel_width):

              #each cell in row
              for cell in numpy.arange(current, (current + kernel_width)):
                # print cell
                nodes_for_fm.append(self.network.layer1_nodes[int(cell)])
              
              #move to next row within kernel
              current+=rtfm1

            #move kernel right
            topleft+=kernel_offset
            current = topleft 

          #move entire kernel down to next row
          topleft+=8
          current = topleft          

        #move down one FM by adding 49
        topleft += 7
        current = topleft

      #prepare to add all feature maps in layer 1 to the NEXT feature map in layer 2
      topleft = 0
      current = topleft

      #can you guys check this function to make sure the shared_weights_2 array is still correct? Todd and I changed this on Sunday.
      #can you also please check if the "adding new node" structure is right (I add once for each FM in layer 2 - so 50 times?)
      #I think we're going to have to redo the forward propagate function based on this new structure
      self.network.layer2_nodes[node].AddConvolutionalInput(nodes_for_fm, shared_weights_2, self.network)


    #is shared_weights_1[fm_cnt] an array of size 9x6? Or of just 9? Should be former. Need to be careful of usage in forward and backprop (change code?)
    #again, does the feature map size variable (i made a new one called fmsize1) already exist? If so, we can use that variable instead of the new one I made.

    
    # connect nodes in layer2 to layer1
    begin = 0
    fm_cnt = 0
    for l2 in range(number_of_layer2_nodes):
      
      current = begin
      nodes_for_fm = []

      for fm in range(number_of_fm_l1): 

        for kernel in range(number_of_nodes_kernel):

          for numrow in range(kernel_width):
            for l1 in numpy.arange(current, (current+kernel_width)): 
              nodes_for_fm.append(self.network.layer1_nodes[l1])
            current += 7

        # end of kernel
        if current % 32 == 0:
          current += 17
        # end of row in kernel
        elif current % fm == 0:
          begin += 10
        else: 
          begin += kernel_offset

      # assign all input nodes and shared weights in this fm for this node
      self.network.layer2_nodes[l2].AddConvolutionalInput \
      (nodes_for_fm, shared_weights_2[fm_cnt], self.network)

      # set begin back to 0 when finished with one feature map
      if (l2+1) % layer2_size == 0 and l2 != 0:
        begin = 0
        fm_cnt += 1
    """

    # connect nodes in hidden layer to layer 2
    for j in range(number_of_hidden_nodes):
      for l2 in range(number_of_layer2_nodes):
        self.network.hidden_nodes[j].AddInput(self.network.layer2_nodes[l2], 0, self.network)              

    # connect nodes in output layer to hidden layer
    for k in range(10):
      for j in range(number_of_hidden_nodes): 
        self.network.outputs[k].AddInput(self.network.hidden_nodes[j], 0, self.network)      

    pass
