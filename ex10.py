import wave, struct
import os,sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import statistics
matplotlib.use( 'tkagg' )

def get_samples(dirpath):
    wavefile = wave.open(dirpath, 'rb')

    nchannels, sampwidth, framerate, nframes, comptype, compname = wavefile.getparams()
    print("nChannels: ", nchannels)
    print("Sampwidth: ", sampwidth)
    print("FrameRate: ", framerate)
    print("nFrames: ", nframes)
    print("CompType: ", comptype)
    print("CompName: ", compname)

    fmt = "<"+'h'*nchannels
    left = []
    right = []

    while wavefile.tell() < nframes:
        dec = struct.unpack(fmt, wavefile.readframes(1))
        left.append(dec[0])
        right.append(dec[1])
        #mean = (dec[0]+dec[1])/2
        #mono.append(mean)
    
    wavefile.close()
    
    return left, right

def probability(samples):
    print("start")
    val, cnts = np.unique(samples, return_counts = True)
    prob = cnts/len(samples)
    return prob

def entropy(prob):
    tmp = 0
    #ordem 1
    for i in range(len(prob)):
        tmp += prob[i]*math.log(prob[i])

    print(tmp)

def main():
    audfile = sys.argv[1]
    order = sys.argv[2]

    files = os.listdir("./WAVfiles")
    print("Files: ", files)
    for filename in files: # loop through all the files and folders

	    if audfile == filename:
	        dir_path = str(os.path.join(os.path.abspath("./WAVfiles"),filename))
        	break
    
    print("Ficheiro a ser lido: ", dir_path)
    left, right = get_samples(dir_path)
    prob_l = probability(left)
    entropy(prob_l)
    prob_r = probability(right)


    

if __name__ == "__main__":
    main()