import wave, struct
import os,sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use( 'tkagg' )




def quantize(file_to_open,nbits):

	waveFile = wave.open(file_to_open, 'rb')

	nchannels, sampwidth, framerate, nframes, comptype, compname = waveFile.getparams()

	print("nChannels: ",nchannels)
	print("Sampwidth: ", sampwidth)
	print("Framerate: ", framerate)
	print("nFrames: ", nframes)
	print("CompType: ", comptype)
	print("CompName: ", compname)

	fmt = "<" + 'h' * nchannels	
	
	mono = []
	xq=[]
	
	while waveFile.tell() < nframes:
		decoded = struct.unpack(fmt, waveFile.readframes(1))
		mean = (decoded[0]+decoded[1])/2
		mono.append(mean)
		

	waveFile.close()

	plt.style.use('seaborn')

	fig, ax = plt.subplots()
	fig1,ax1 = plt.subplots()
	maxs = max(mono)
	mins = min(mono)

	t=np.linspace(0, len(mono)/framerate, num=len(mono))

	q=math.pow(2,8) # 8 bit quantizer

	b = int(nbits)           # number of bits for quantization. 
	levels = 2**b         # 2 bits equal 4 quantization levels
	q = math.floor((maxs-mins)/levels)
	A = max(maxs,abs(mins))
	lq = np.linspace(mins,maxs,num=levels)
	print("levels: ",levels)
	print("q: ", q)
	print("lq: ", lq)	

	for i in range(len(mono)):
		if mono[i]<lq[0]:
			tmp=lq[0]

		elif mono[i] > lq[-1]:
			tmp=lq[-1]

		else:
		    idx = (np.abs(lq - mono[i])).argmin()
		    tmp=lq[idx]

		
		xq.append(tmp)
	
	
	print("\nMaxMono: ", maxs)
	print("MinMono: ", mins)	
	print("\nMaxQuantized: ",max(xq))
	print("Min Quantized: ", min(xq))

	ax.plot(t,mono,label='mono')
	ax1.plot(t,xq,label='quantized')
	ax.set_title("Mono")
	ax1.set_title("quantized")

	plt.show()


def main():


	open_file = sys.argv[1]
	nbits = sys.argv[2]
 

	filenames= os.listdir ("./WAVfiles") # get all files' and folders' names in the current directory
	print("Filenames: ", filenames)

	for filename in filenames: # loop through all the files and folders

	    if open_file == filename:
	        dir_path = str(os.path.join(os.path.abspath("./WAVfiles"),filename))
        	break

	print("Ficheiro a ser lido: ", dir_path)
	quantize(dir_path,nbits)

if __name__ == "__main__":

    main()