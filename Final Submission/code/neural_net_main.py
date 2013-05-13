from data_reader import *
from neural_net import *
from neural_net_impl import *
import sys
import random
import py.test
import numpy # added to do arange

def parseArgs(args):
  """Parses arguments vector, looking for switches of the form -key {optional value}.
  For example:
    parseArgs([ 'main.py', '-e', 20, '-r', 0.1, '-m', 'Simple' ]) = { '-e':20, '-r':5, '-t': 'simple' }"""
  args_map = {}
  curkey = None
  for i in xrange(1, len(args)):
    if args[i][0] == '-':
      args_map[args[i]] = True
      curkey = args[i]
    else:
      assert curkey
      args_map[curkey] = args[i]
      curkey = None
  return args_map

def validateInput(args):
  args_map = parseArgs(args)
  assert '-e' in args_map, "A number of epochs should be specified with the flag -e (ex: -e 10)"
  assert '-r' in args_map, "A learning rate should be specified with the flag -r (ex: -r 0.1)"
  assert '-t' in args_map, "A network type should be provided. Options are: simple | hidden | custom"
  assert '-m' in args_map, "A mode should be specified with the flag -m. Options are: train | run | test"

  return(args_map)

def main():
  # Parsing command line arguments
  args_map = validateInput(sys.argv)
  epochs = int(args_map['-e'])
  rate = float(args_map['-r'])
  networkType = args_map['-t']
  modeType = args_map['-m']

  # Load in the training data.
  images = DataReader.GetImages('BWtraining.txt', -1)
  for image in images:
    assert len(image.pixels) == 14
    assert len(image.pixels[0]) == 14

  # Load the validation set.
  validation = DataReader.GetImages('validation-1k.txt', -1)
  for image in validation:
    assert len(image.pixels) == 14
    assert len(image.pixels[0]) == 14

  # Initializing network

  if networkType == 'simple':
    network = SimpleNetwork()
  if networkType == 'hidden':
    network = HiddenNetwork()
  if networkType == 'custom':
    network = CustomNetwork()
    
  # Hooks user-implemented functions to network
  network.FeedForwardFn = FeedForward
  network.TrainFn = Train


  # Displays information
  print '* * * * * * * * *'
  print 'Parameters => Epochs: %d, Learning Rate: %f' % (epochs, rate)
  print 'Type of network used: %s' % network.__class__.__name__
  print 'Running mode: %s' % modeType
  print ('Input Nodes: %d, Hidden Nodes: %d, Output Nodes: %d' %
         (len(network.network.inputs), len(network.network.hidden_nodes),
          len(network.network.outputs)))
  print '* * * * * * * * *'


  # Initialize network weights and train, run or test the network.
  if modeType == 'train':
    network.InitializeWeights()
    network.Train(images, validation, rate, epochs, networkType)
    DataReader.DumpWeights(network.network.weights, 'g_weights')

  if modeType == 'run':
    network.InitializeReadWeights('g_weights')
    network.ReadWeightsPerf(images, validation, epochs, networkType)

  if modeType == 'test': 
    assert '-i' in args_map, "An increment number should be specified with the flag -i for the learning rate"
    assert '-n' in args_map, "A number of times should be specified with the flag -n for how many times the \
                             learning rate should increase by the -i increment"
    inc = float(args_map['-i'])
    rate_times = int(args_map['-n'])   
    for i in range(rate_times):
      network.InitializeWeights()
      network.Train(images, validation, rate, epochs, networkType)
      rate += inc 

if __name__ == "__main__":
  main()
