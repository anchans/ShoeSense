import socket
import struct 
import matplotlib.pyplot as plt
from sys import argv
import signal
import sys
import subprocess


UDP_IP = "128.237.164.59"
UDP_PORT = 7123

script, filename = argv

print filename

target = open(filename, 'w')
def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
	target.flush()
	target.close()
	#subprocess.call(['python', 'readnPlot.py', filename])
	sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


sock = socket.socket(socket.AF_INET, # Internet
                       socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))

while True:
	data, addr = sock.recvfrom(30)
	
	#print int.from_bytes(b(data))
	n = struct.unpack(">L", data[8:12])[0]
	f = struct.unpack(">L", data[0:4])[0]
	b = struct.unpack(">L", data[4:8])[0]
	print "{} {} {}".format(n, f, b)
	target.write(str(n)+" "+str(f)+" "+str(b)+"\n")

