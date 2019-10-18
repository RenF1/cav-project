import wave, struct
#from scipy.stats import norm
import os,sys
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
#import scipy.misc
matplotlib.use( 'tkagg' )
		
def calc_snr(org, quant):
	row = org.shape[0]
	col = org.shape[1]
	mn = 0
	tmp = 0
	snr = 0
	for i in range(row-1):
		for j in range(col-1):
			tmp += math.pow(abs(org[i, j]-quant[i, j]), 2)
			mn += math.pow(org[i, j], 2)
	
	var = tmp/(row*col)
	mn = mn/(row*col)

	snr = 10*(math.log(mn/var)/math.log(10))
	return snr



def split_n_calc(org, quant):
	bo, go, ro = cv2.split(org)
	bq, gq, rq = cv2.split(quant)
	
	snrb = calc_snr(bo, bq)
	snrg = calc_snr(go, gq)
	snrr = calc_snr(ro, rq)

	return snrb, snrg, snrr

def read_vid(org, quant):
	cap1 = cv2.VideoCapture(org)
	cap2 = cv2.VideoCapture(quant)
	snrb = []
	snrg = []
	snrr = []
	while(1):
		ret1, frame1 = cap1.read()
		ret2, frame2 = cap2.read()

		if ret1 == 1 and ret2 == 1:
			sb, sg, sr = split_n_calc(frame1, frame2)
			#print("BLUE, GREEN, RED", sb, sg, sr)
			snrb.append(sb)
			snrg.append(sg)
			snrr.append(sr)
		else:
			print("Video ended!")
			break
	snrb  = np.mean(snrb)
	snrg  = np.mean(snrg)
	snrr  = np.mean(snrr)
	cap1.release()
	cap2.release()
	return snrb, snrg, snrr

def main():


	mode = sys.argv[1]
	
	if mode == 'v':
		filenames= os.listdir ("./videos") # get all files' and folders' names in the current directory
		print("Videos: ", filenames)
		open_vid = str(input("Choose the video: "))
		for  filename in filenames:
			if open_vid == filename:
				path_or = str(os.path.join(os.path.abspath("./videos"),filename))
				break
		quantizefiles = os.listdir("./output")
		print("Quantize Files: ", quantizefiles)
		quant_vid = str(input("Choose the image: "))
		for filename in quantizefiles:
			if filename == filename:
				path_qt = str(os.path.join(os.path.abspath("./output"),filename))
				break
		
		snrb, snrg, snrr = read_vid(path_or, path_qt)
		print("BLUE SNR:", snrb)
		print("GREEN SNR:", snrg)
		print("RED SNR:",  snrr)
		

	elif mode == 'i':

		filenames= os.listdir ("./img_dataset") # get all files' and folders' names in the current directory
		print("Filenames: ", filenames)
		open_img = str(input("Choose the image: "))
		for  filename in filenames:
			if open_img == filename:
				path_or = str(os.path.join(os.path.abspath("./img_dataset"),filename))
				break


		quantizefiles = os.listdir("./output")
		print("Quantize Files: ", quantizefiles)
		quant_img = str(input("Choose the image: "))
		for filename in quantizefiles:
			if quant_img == filename:
				path_qt = str(os.path.join(os.path.abspath("./output"),filename))
				break
		img1 = cv2.imread(path_or, 1)
		img2 = cv2.imread(path_qt, 1)
		snrb, snrg, snrr = split_n_calc(img1, img2)
		print("BLUE SNR:", snrb)
		print("GREEN SNR:", snrg)
		print("RED SNR:",  snrr)


if __name__ == "__main__":

    main()