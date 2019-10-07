import wave, struct
import os,sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use( 'tkagg' )

def getHist(file_to_open):

	waveFile = wave.open(file_to_open, 'rb')


	nchannels, sampwidth, framerate, nframes, comptype, compname = waveFile.getparams()


	print("nChannels: ",nchannels)
	print("Sampwidth: ", sampwidth)
	print("Framerate: ", framerate)
	print("nFrames: ", nframes)
	print("CompType: ", comptype)
	print("CompName: ", compname)

	fmt = "<" + 'h' * nchannels	
	left = []
	right = []
	mono = []
	
	while waveFile.tell() < nframes:
		decoded = struct.unpack(fmt, waveFile.readframes(1))
		mean = (decoded[0]+decoded[1])/2
		left.append(decoded[0])
		right.append(decoded[1])
		mono.append(mean)
		

	waveFile.close()

	plt.style.use('seaborn')

	fig, ax = plt.subplots(1, 3)
	ax = ax.ravel()

	t=np.linspace(0, len(mono)/framerate, num=len(mono))
	
	maxy=[]
	miny=[]
	titles=('Left','Right','Mono')

	#(nL, binsL, patchL) = plt.hist(left,bins=1000)
	#(nR,binsR,patchR) = plt.hist(right,bins=1000)
	#(nM,binsM,patchM)= plt.hist(mono,bins=1000)
	
	maxy.append(max(left))
	maxy.append(max(right))
	maxy.append(max(mono))

	miny.append(min(left))
	miny.append(min(right))
	miny.append(min(mono))

	print("MaxLeft: ", maxy[0])
	print("MinLeft: ", miny[0])
	print("\nMaxRight: ", maxy[1])
	print("MinRight: ", miny[1])
	print("\nMaxMono: ", maxy[2])
	print("MinMono: ", miny[2])

	# Creating an empty dict 
	#n = list()
	#n.append(nL)
	#n.append(nR)
	#n.append(nM)

	#bins=list()
	#bins.append(np.linspace(miny[0],min)

	data = list()
	data.append(left)
	data.append(right)
	data.append(mono)
	

	for idx,a in enumerate(ax):
	    a.hist(data[idx],bins=np.linspace(miny[idx],maxy[idx],150))
	    a.set_title(titles[idx])
	    

	plt.suptitle('Left|Right|Mono histograms', y=1.05, size=16)
	fig.tight_layout()
	
	f2 = plt.figure(2)
	plt.title("Signal wave")
	plt.plot(t,mono)
	plt.xlabel('Time')


	plt.show()


def main():


	open_file = sys.argv[1]
 

	filenames= os.listdir ("./WAVfiles") # get all files' and folders' names in the current directory
	print("Filenames: ", filenames)

	for filename in filenames: # loop through all the files and folders

	    if open_file == filename:
	        dir_path = str(os.path.join(os.path.abspath("./WAVfiles"),filename))
        	break

	print("Ficheiro a ser lido: ", dir_path)
	getHist(dir_path)

if __name__ == "__main__":

    main()