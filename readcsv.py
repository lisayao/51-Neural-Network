### Read in CSV file and make it into a vector

import csv

with open('train.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=' ')
	for row in reader:
		vector = row
