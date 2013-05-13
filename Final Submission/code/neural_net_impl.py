from neural_net import *
import random
import numpy # added to do arange
import math
import sys
import pytest
import py.test
from data_reader import *

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

  """
  network.CheckComplete()


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

  """

  network.CheckComplete()

  # 1) We first propagate the input through the network
  # 2) Then we compute the errors and update the weights starting with the last layer
  # 3) We now propagate the errors to the hidden layer, and update the weights there too

  FeedForward(network, input, network_type)

  number_of_fm_l1 = 6
  number_of_fm_l2 = 50
  kernel_size = 9


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

    layer2_size = 9
    number_of_fm_l1 = 6
    kernel_width = 3
    kernel_offset = 2

    begin = 0
    cnt = 0
    weight_index = 0
    for l2 in range(layer2_length):
      weight_index = 0
      current = begin

      # get a single kernel from each fm
      for fm in range(number_of_fm_l1): 
        # get kernel in current fm
        for numrow in range(kernel_width):
          for l1 in numpy.arange(current, (current+kernel_width)): 
            sum[l1] += network.layer2_nodes[l2].weights[weight_index].value * error_l2[l2]
            weight_index += 1
          current += 7

        # move to next fm
        current += 28

      begin += kernel_offset
      if begin == 279:
        begin = 0
        cnt = 0
      elif begin == (6 + 14 * cnt):
        begin += 8
        cnt += 1

      # if at end of fm in layer 2 then reset begin to zero
      if (l2+1) % layer2_size == 0 and l2 != 0:
        begin = 0
        cnt = 0
    
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

    index = 0
    lst = [0, 2, 4, 14, 16, 18, 28, 30, 32]

    for subarray in shared_weights_2:
      wcnt = 0
      for weight in subarray:
        l2 = index      
        for l1 in lst:
          weight.value += learning_rate * network.layer1_nodes[l1].transformed_value * error_l2[l2]
          l2 += 1
        wcnt += 1
        if (wcnt == 3) or (wcnt == 6):
          lst = map(lambda x: x + 5, lst)
        elif (wcnt == 9):
          lst = map(lambda x: x + 33, lst)
          wcnt = 0
        else:
          lst = map(lambda x: x + 1, lst)
      lst = [0, 2, 4, 14, 16, 18, 28, 30, 32]
      index += 9


    #backprop to update weights of layer 0 to 1
    index = 0
    lst = [0, 2, 4, 6, 8, 10, 12, 30,32,34,36,38,40,42, 60,62,64,66,68,70,72, 90,92,94,96,98,100,102,120, \
          122,124,126,128,130,132, 150,152,154,156,158,160,162, 180,182,184,186,188,190,192]
    
    for subarray in shared_weights_1:
      wcnt = 0      
      for weight in subarray:
        l1 = index        
        for l0 in lst:      
          weight.value += learning_rate * network.inputs[l0].transformed_value*error_l1[l1]          
          l1 += 1
        wcnt += 1
        if (wcnt == 3) or (wcnt == 6):
          lst = map(lambda x: x + 13, lst)        
        else:
          lst = map(lambda x: x + 1, lst)
      lst = [0, 2, 4, 6, 8, 10, 12, 30,32,34,36,38,40,42, 60,62,64,66,68,70,72, 90,92,94,96,98,100,102,120, \
            122,124,126,128,130,132, 150,152,154,156,158,160,162, 180,182,184,186,188,190,192]
      index += 49

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

  # for e in range(epochs):
  input_length = len(inputs)
  for i in range(input_length):
    Backprop(network, inputs[i], targets[i], learning_rate, network_type)
    

  network.CheckComplete()
  pass    


class EncodedNetworkFramework(NetworkFramework):
  def __init__(self):
    
    super(EncodedNetworkFramework, self).__init__() 



    # Trying to make function to work with ReadWeights.

  def ReadWeightsPerf(self, images, validation_images, epochs, network_type):

    # Convert the images and labels into a format the network can understand.
    inputs = []
    targets = []
    for image in images:
      inputs.append(self.Convert(image))
      targets.append(self.EncodeLabel(image.label))

    for i in range(epochs):

      # Print out the current training and validation performance.
      perf_train = self.Performance(images, network_type)
      perf_validate = self.Performance(validation_images, network_type)
        
      print '%d Performance: %.8f %.3f' % (
        i + 1, perf_train, perf_validate)  

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
    
    """

    for weight in self.network.weights:
      weight.value = random.uniform(-0.01,0.01)


  # new InitWeights function for reading weights

  def InitializeReadWeights(self, filename):
    
    weights2 = []
    weights2 = DataReader.ReadWeights(filename)
    

    for weight in range(len(self.network.weights)):
      self.network.weights[weight].value = weights2[weight]    



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
   
    
    super(CustomNetwork, self).__init__() 

    ### Create proper size for shared weight globals ###
    kernel_size = kernel_width * kernel_width

    layer1_width = int(math.ceil((15 - kernel_width + 1.) / kernel_offset))
    assert layer1_width == 7, '1st layer must be 7x7'
    layer1_size = layer1_width * layer1_width

    layer2_width = int(math.ceil((layer1_width - kernel_width + 1.) / kernel_offset))
    assert layer2_width == 3, '2nd layer must be 3x3'
    layer2_size = layer2_width * layer2_width


    global shared_weights_1
    shared_weights_1 = []
    for i in range(number_of_fm_l1):
      subarray = []
      for j in range(kernel_size):
        weight = self.network.GetNewWeight()
        subarray.append(weight)
      shared_weights_1.append(subarray)

    global shared_weights_2
    shared_weights_2 = []
    for i in range(number_of_fm_l2):
      subarray = []
      for j in range(kernel_size*number_of_fm_l1):
        weight = self.network.GetNewWeight()
        subarray.append(weight)
      shared_weights_2.append(subarray)

    for i in range(number_of_fm_l1):
      for j in range(kernel_size):
        shared_weights_1[i][j].value = random.uniform(-0.01,0.01)
    
    for i in range(number_of_fm_l2):
      for j in range(kernel_size):
        shared_weights_2[i][j].value = random.uniform(-0.01,0.01)
    


    ### Add appropriate number of nodes for each layer ###

    # 1) Adds an input node for each pixel
    for i in range(225): # using 15x15
      self.network.AddNode(Node(), 1)

    # 2) Adds layer 1 (1st convolution)
    number_of_layer1_nodes = layer1_size * number_of_fm_l1

    for l1 in range(number_of_layer1_nodes):
      self.network.AddNode(Node(), 2) 

    # 2) Adds layer 2 (2nd convolution)
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

    # connect nodes in layer1 to input layer
    begin = 0
    cnt = 0
    fm_cnt = 0
    for l1 in range(number_of_layer1_nodes):
      current = begin
      inputs_for_cur_node = []
      for numrow in range(kernel_width):
        for i in numpy.arange(current, (current+kernel_width)): 
          inputs_for_cur_node.append(self.network.inputs[i])          
        current += 15

      # assign all input nodes and shared weights in this fm for this node
      self.network.layer1_nodes[l1].AddConvolutionalInput(inputs_for_cur_node, \
        shared_weights_1[fm_cnt], self.network)

      begin += kernel_offset
      if begin == 194:
        begin = 0
        cnt = 0
      elif begin == (14 + 30 * cnt):
        begin += 16
        cnt += 1

      # if at end of fm in layer 1 then reset begin to zero
      if (l1+1) % layer1_size == 0 and l1 != 0:
        begin = 0
        fm_cnt += 1

    # connect nodes in layer2 to layer1
    begin = 0
    cnt = 0
    fm_cnt = 0
    for l2 in range(number_of_layer2_nodes):
      current = begin
      inputs_for_cur_node = []

      # get a single kernel from each fm
      for fm in range(number_of_fm_l1): 
        # get kernel in current fm
        for numrow in range(kernel_width):
          for l1 in numpy.arange(current, (current+kernel_width)): 
            inputs_for_cur_node.append(self.network.layer1_nodes[l1])            
          current += 7

        # move to next fm
        current += 28

      assert len(inputs_for_cur_node) == 54

      # assign all input nodes and shared weights in all fm_l1 for this node
      self.network.layer2_nodes[l2].AddConvolutionalInput(inputs_for_cur_node, \
        shared_weights_2[fm_cnt], self.network)

      begin += kernel_offset
      if begin == 279:
        begin = 0
        cnt = 0
      elif begin == (6 + 14 * cnt):
        begin += 8
        cnt += 1

      # if at end of fm in layer 2 then reset begin to zero
      if (l2+1) % layer2_size == 0 and l2 != 0:
        begin = 0
        cnt = 0
        fm_cnt += 1

    # connect nodes in hidden layer to layer 2
    for j in range(number_of_hidden_nodes):
      for l2 in range(number_of_layer2_nodes):
        self.network.hidden_nodes[j].AddInput(self.network.layer2_nodes[l2], 0, self.network)              

    # connect nodes in output layer to hidden layer
    for k in range(10):
      for j in range(number_of_hidden_nodes): 
        self.network.outputs[k].AddInput(self.network.hidden_nodes[j], 0, self.network)      

    pass
