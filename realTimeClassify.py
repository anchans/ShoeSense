import os
import signal
import sys
import socket
import struct 
import subprocess
import numpy as np
from sys import argv
import matplotlib.pyplot as plt
import computeFeatures
import threading
import Queue
import signal

UDP_IP = "128.237.212.124"
UDP_PORT = 7123
window = 300

def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    #	t.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

q = Queue.Queue()

def worker():
    while True:
        item = q.get()
        if item is None:
            break
        computeFeatures.main(item[0], item[1])
        q.task_done()
        #print 'done'

def get_samples(n):
	i = 0
	front = list()
	back = list()
	for x in xrange(n):
		data, addr = sock.recvfrom(30)
		n = struct.unpack(">L", data[8:12])[0]
		f = struct.unpack(">L", data[0:4])[0]
		b = struct.unpack(">L", data[4:8])[0]
		front.append(int(f))
		back.append(int(b))
		#if x%50 == 0:
		#	print "."

	return front, back


t = threading.Thread(target=worker)
t.start()

while True:
	(front, back) = get_samples(window)
	q.put([front, back])
	#print "push"
	

