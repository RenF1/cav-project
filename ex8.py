import wave, struct
import os,sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#import statistics
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
	print("levels: ",levels)
	print("q: ", q)	

	for i in range(len(mono)):
		tmp = math.floor((mono[i]-mins)/q)
		tmp = tmp*q+(q/2)+mins

		if tmp<mins:
			tmp=mins+(q/2)

		elif tmp>maxs:
			tmp=maxs-(q/2)
		
		xq.append(tmp)
	
	return mono, xq
def snr_calc(orig, xq):
	tmp = 0
	mn = 0
	for i in range(len(xq)):
		tmp += math.pow(abs(xq[i]-orig[i]), 2)
		mn += math.pow(orig[i], 2)

	var = tmp/len(xq)
	mn = mn/len(orig)

	snr = 10*(math.log(mn/var)/math.log(10))
	return snr

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
	mono, xq = quantize(dir_path,nbits)

	snr = snr_calc(mono, xq)
	print("Media: ", np.mean(mono))
	print("Variancia: ", snr) 
	print("thats all")


if __name__ == "__main__":

    main()