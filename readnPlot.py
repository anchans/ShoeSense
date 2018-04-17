
import matplotlib.pyplot as plt
from sys import argv

script, filename = argv

file = open(filename, "r")
array = file.readlines()

front = []
back = []
avg = []
for i in range(len(array)):
	vals = array[i].split(" ")
	front.append(vals[1])
	back.append(vals[2].rstrip('\n'))

for i in range(len(front)):
	avg.append((int(front[i]) + int(back[i])) / 2);

plt.plot(avg)
plt.xlabel('Samples')
plt.ylabel('Average of Sensor Values')
plt.savefig(filename + '.jpeg')
plt.show()
