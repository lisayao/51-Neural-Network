from neural_net import *
import random
import numpy
import math
import py.test
from data_reader import *
from neural_net_impl import *
import fileinput, sys




network = HiddenNetwork()
network.FeedForwardFn = FeedForward


outfile = open("BWtraining.txt", 'w')

images = DataReader.GetImages("training-9k.txt", "5")
i = 0
for image in images:
  outfile.write('#%s \n' % image.label)
  for pixel in (network.Convert(image)):
	 if i == 0:
	 	if pixel * 256 > 90:
	 	  outfile.write('  %s   ' % (str(int(255.))))
	 	else:
	 	  outfile.write('  %s   ' % (str(int(0.))))	
	 	i+=1
	 elif i==13:
	 	if pixel * 256 > 90:
	 	  outfile.write('%s\n' % (str(int(255.))))
	 	else:
	 	  outfile.write('%s\n' % (str(int(0.))))
	  	i = 0
	 else:
	 	if pixel * 256 > 90:
		  outfile.write('%s   ' % (str(int(255.))))
		else:  
		  outfile.write('%s   ' % (str(int(0.0))))	
		i += 1










