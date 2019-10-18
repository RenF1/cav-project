import wave, struct
import os,sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
#import statistics
matplotlib.use( 'tkagg' )

def get_samples(dirpath):

    samples=np.array([])
    filenames= os.listdir ("./videos") # get all files' and folders' names in the current directory
    print("Videos: ", filenames)
    #open_vid = str(input("Choose the video: "))
    open_vid = "football_qcif_15fps.y4m"

    for filename in filenames: # loop through all the files and folders

        if open_vid == filename:
            dir_path = str(os.path.join(os.path.abspath("./videos"),filename))
            break


    # Create a VideoCapture object and read from input file 
    cap = cv2.VideoCapture(dir_path) 
       
    # Check if camera opened successfully 
    if (cap.isOpened()== False):  
      print("Error opening video file") 
       
    # Read until video is completed 
    while(cap.isOpened()): 
          
        # Capture frame-by-frame 
        ret, frame = cap.read()

        if ret == True:
            # Display the resulting frame 
            samples=np.concatenate((samples,frame.flatten()),axis=None)

            # Press Q on keyboard to  exit 
            if cv2.waitKey(33) & 0xFF == ord('q'): 
                break
           
        # Break the loop 
        else:  
            break

    cap.release()

    return samples

def twos_comp(val, bits):
    """compute the 2's complement of int value val"""
    if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val                         # return positive value as is

def probability(samples):
    print("start")
    first=1
    count0=0.0
    count1=0.0
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
        nbin_2c=twos_comp(int(samples[x]), 7)
        sbin='{:016b}'.format(nbin_2c)

        for y in range(len(sbin)):
            if y==0:
                comp=0
            if sbin[y]!='-':
                if y==comp and first==0:
                    bd1=bd2
                    bd2=sbin[y]
                    if bd1=='0':
                        count0+=1
                    else:
                        count1+=1    
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
                    if bd1=='0':
                        count0+=1
                    else:
                        count1+=1    
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

    prob=[count00/(16*len(samples)), count01/(16*len(samples)), count10/(16*len(samples)), count11/(16*len(samples)), count0/(16*len(samples)), count1/(16*len(samples))]
    P0_0=prob[0]/prob[4] #conditional probabilities
    P0_1=prob[2]/prob[5]
    P1_0=prob[1]/prob[4]
    P1_1=prob[3]/prob[5]
    probs_cond=[P0_0,P0_1,P1_0,P1_1]
    #print(P0_0)
    #print(P0_1)
    #print(P1_0)
    #print(P1_1)
    return probs_cond

def entropy(prob):

    PS0=prob[1]/(prob[1]+prob[2])
    PS1=prob[2]/(prob[1]+prob[2])
    HS1=-prob[1]*(math.log(prob[1],2))-prob[3]*(math.log(prob[3],2))
    HS0=-prob[2]*(math.log(prob[2],2))-prob[0]*(math.log(prob[0],2))
    H=PS0*HS0+PS1*HS1
    print(" ")
    print("     Entropy: "),
    print(H),
    print("bits")

def main():
    audfile = sys.argv[1]

    files = os.listdir("./videos")
    print("Files: ", files)
    for filename in files: # loop through all the files and folders

	    if audfile == filename:
	        dir_path = str(os.path.join(os.path.abspath("./videos"),filename))
        	break
    
    print("Ficheiro a ser lido: ", dir_path)
    smps=get_samples(dir_path)
    prob = probability(smps)
    entropy(prob)

if __name__ == "__main__":
    main()