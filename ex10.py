import wave, struct
import os,sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#import statistics
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
    
    wavefile.close()
    
    return left, right

def twos_comp(val, bits):
    """compute the 2's complement of int value val"""
    if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val                         # return positive value as is

def probability(samples):
    print("start")
    first=1
    count00=0.0
    count01=0.0
    count10=0.0
    count11=0.0
    comp=0
    for x in range(len(samples)):
        percentage=(float(x)/len(samples))*100
        print("Progress: "),
        print("{:.1f}".format(percentage)),
        print("%")
        nbin_2c=twos_comp(samples[x], 15);
        sbin='{:016b}'.format(nbin_2c)
        for y in range(len(sbin)):
            if y==0:
                comp=0
            if sbin[y]!='-':
                if y==comp and first==0:
                    bd1=bd2
                    bd2=sbin[y]
                    if bd1=='0'and bd2=='0': 
                        count00+=1
                    if bd1=='0'and bd2=='1': 
                        count01+=1
                    if bd1=='1'and bd2=='0': 
                        count10+=1
                    if bd1=='1'and bd2=='1': 
                        count11+=1

                if y<len(sbin)-1:
                    bd1=sbin[y] #binary digit 1
                    bd2=sbin[y+1] #binary digit 2
                    if bd1=='0'and bd2=='0': 
                        count00+=1
                    if bd1=='0'and bd2=='1': 
                        count01+=1
                    if bd1=='1'and bd2=='0': 
                        count10+=1
                    if bd1=='1'and bd2=='1': 
                        count11+=1
            else:
                comp=1
        first=0

    print(count00)
    print(count01)
    print(count10)
    print(count11)
    prob=[count00/(16*len(samples)), count01/(16*len(samples)), count10/(16*len(samples)), count11/(16*len(samples))]
    print(prob[0])
    print(prob[1])
    print(prob[2])
    print(prob[3])

    return prob

def entropy(prob):
    PS0=prob[1]/(prob[1]+prob[2])
    PS1=prob[2]/(prob[1]+prob[2])
    HS1=-prob[1]*(math.log(prob[1],2))-prob[3]*(math.log(prob[3],2))
    HS0=-prob[2]*(math.log(prob[2],2))-prob[0]*(math.log(prob[0],2))
    H=PS0*HS0+PS1*HS1
    print(" ")
    print("Entropy: "),
    print(H)

def main():
    audfile = sys.argv[1]

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
    #prob_r = probability(right)
    #entropy(prob_r)

    

if __name__ == "__main__":
    main()