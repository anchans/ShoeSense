import os
import numpy as np
from sys import argv
import computeFeatures

window = 250
prevBuffer = 180

files = list()
folders = ['data_Advait/experiments', 'data_Artur/experiments']

for folder in folders:
    for file_name in os.listdir(folder):
        files.append(folder+'/'+file_name)

for i in range(len(files)):

    print ""
    print "***" + files[i] + " *****"
    print ""

    file = open(files[i], "r")
    lines = file.readlines()
    front = list()
    back = list()

    for i in range(len(lines)):
		vals = lines[i].split(" ")
		front.append(int(vals[1]))
		back.append(int(vals[2].rstrip('\n')))

		if (len(front) > window):
			computeFeatures.main(front, back)
			del front[:prevBuffer]
			del back[:prevBuffer]
	    
    file.close()

    print ""
    print "************************"

