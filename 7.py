import wave, struct
from scipy.stats import norm
import os,sys
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use( 'tkagg' )


def plotPDF(file_path):

	img = cv2.imread(file_path,1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	plt.style.use('seaborn')
	
	(b,g,r)=cv2.split(img)

	sigR = round(np.std(r),2)
	sigG = round(np.std(g),2)
	sigB = round(np.std(b),2)
	sigGray = round(np.std(gray),2)

	print("Sigma R: ", sigR)
	print("Sigma G; ", sigG)
	print("Sigma B: ", sigB)
	print("Sigma Gray: ", sigGray)

	meanR = round(np.mean(r),2)
	meanG = round(np.mean(g),2)
	meanB = round(np.mean(b),2)
	meanGray = round(np.mean(gray),2)

	print("Mean R: ", meanR)
	print("Mean G; ", meanG)
	print("Mean B: ", meanB)
	print("Mean Gray: ", meanGray)
	

	histR,binsR = np.histogram(r,bins=np.max(r)-np.min(r),range=(np.min(r),np.max(r)),density=True)
	histG,binsG = np.histogram(g,bins=np.max(g)-np.min(g),range=(np.min(g),np.max(g)),density=True)
	histB,binsB = np.histogram(b,bins=np.max(b)-np.min(b),range=(np.min(b),np.max(b)),density=True)
	hist,bins = np.histogram(gray,bins=np.max(gray)-np.min(gray),range=(np.min(gray),np.max(gray)),density=True)

	maxR = max(histR)
	maxG = max(histG)
	maxB = max(histB)
	maxGr = max(hist)
	x = max(*[ maxR, maxG, maxB]) 

	binsR = np.max(r)-np.min(r)
	binsG = np.max(g)-np.min(g)
	binsB = np.max(b)-np.min(b)
	bins = np.max(gray)-np.min(gray)

	fig,ax=plt.subplots(1,2)

	lineR, = ax[0].plot(np.arange(binsR), np.zeros((binsR,)), c='r',label='Red') 
	lineG, = ax[0].plot(np.arange(binsG), np.zeros((binsG,)), c='g',label='Green')
	lineB, = ax[0].plot(np.arange(binsB), np.zeros((binsB,)), c='b',label='Blue')
	lineGray, = ax[1].plot(np.arange(bins), np.zeros((bins,1)), c='k', label='intensity')

	
	lineR.set_ydata(histR)
	lineG.set_ydata(histG)
	lineB.set_ydata(histB)
	lineGray.set_ydata(hist)

	ax[0].set_ylim(0,x+0.05)
	ax[0].set_xlim(0,bins-1)
	ax[1].set_ylim(0,max(hist)+0.05)
	ax[1].set_xlim(0,bins-1)
	ax[0].legend()
	ax[1].legend()

	plt.suptitle('PDF function', y=1.05, size=16)

	path = os.getcwd()+'/output/pdf_plot.png'
	plt.savefig(path)

	plt.show()


def lloydAlgorithm(image_path,nbits,iter,epsilon=0.01):

	img = cv2.imread(image_path,1)
	(b,g,r)=cv2.split(img)

	levelsR = randomInit(r,nbits)
	levelsG = randomInit(g,nbits)
	levelsB = randomInit(b,nbits)

	print("levelsR: ", levelsR)
	print("levelsG: ", levelsG)
	print("levelsB: ",levelsB)

	for i in range(0,iter):

		bkR = getBorders(levelsR)
		bkG = getBorders(levelsG)
		bkB = getBorders(levelsB)

		print("bkR: ", bkR)
		print("bkG: ",bkG)
		print("bkB: ",bkB)

		ykR = reconstructValues(bkR)
		ykG = reconstructValues(bkG)
		ykB = reconstructValues(bkB)

		print("ykR: ",ykR)
		print("ykG: ",ykG)
		print("ykB: ",ykB)

		xqR = quantize(r,bkR,ykR)
		xqG = quantize(g,bkG,ykG)
		xqB = quantize(b,bkB,ykB)

		epsR = getError(r,nbits)
		epsG = getError(g,xqG,nbits)
		epsB = getError(b,xqB,nbits)

		if epsilon>epsR:
			minbk = bkR
			minyk = ykR


	image = cv2.merge((ykB,ykG,ykR))
	while(1):
		cv2.imshow("Quantized Lloyd",image)
		if cv2.waitKey(1) & 0xFF == ord('q'): 
			break

	
def randomInit(x,nbits):

	q = np.power(2,nbits)
	return np.linspace(np.min(x),np.max(x),num=q)

def getBorders(x):

	bk = np.zeros(len(x)-1)

	for i in range(len(x)-1):
		bk[i]=0.5*(x[i]+x[i+1])

	return bk

def reconstructValues(x):

	xq = np.zeros(len(x)-1)
	#hist,bins = np.histogram(x,bins=np.max(int(x))-np.min(int(x)),range=(np.min(x),np.max(x)),density=True)

	for i in range(len(x)-1):
		dx = (x[i+1]-x[i])/100
		f = np.arange(x[i],x[i+1],dx)		
		xq[i]=(x[i]*sum(f)*dx)/(sum(f)*dx)

	return xq


def quantize(x,b,y):
	
	xq = np.zeros(np.shape(x))
	print("sizeB: ",np.size(b))
	print("sizeY: ",np.size(y))
	for i in range(len(b)-1):

		if i==0:

			xq = np.where(np.logical_and(x > b[i],x <= b[i+1]),np.full(np.size(xq),y[i]),xq)

		elif i == range(len(b))[-1]-1:
			xq = np.where(np.logical_and(x > b[i], x <= b[i+1]),np.full(np.size(xq), y[i]), xq)

		else:
			xq = np.where(np.logical_and(x > b[i], x < b[i+1]), np.full(np.size(xq), y[i]), xq)

		return xq

def getError(x,xq,nbits):

	assert np.size(x) == np.size(xq)

	return np.sum(np.power(x-xq,2))/np.size(x)









def main():


	mode = sys.argv[1]
	nbits=int(sys.argv[2])
	iterations=int(sys.argv[3])
	#mode = i/v (image or video)


	#while mode!= 'i' or mode!= 'v':
	#	print("Press i or v")
	#	option = int(input("Choose again: "))

	if mode == 'v':
		filenames= os.listdir ("./vídeos") # get all files' and folders' names in the current directory
		print("Videos: ", filenames)
		open_vid = str(input("Choose the video: "))

		for filename in filenames: # loop through all the files and folders

		    if open_vid == filename:
		        dir_path = str(os.path.join(os.path.abspath("./vídeos"),filename))
	        	break

		getHistVideo(dir_path)

	elif mode == 'i':

		filenames= os.listdir ("./img_dataset") # get all files' and folders' names in the current directory
		print("Filenames: ", filenames)
		open_img = str(input("Choose the image: "))

		for filename in filenames: # loop through all the files and folders

		    if open_img == filename:
		        dir_path = str(os.path.join(os.path.abspath("./img_dataset"),filename))
	        	break

		#plotPDF(dir_path)
		lloydAlgorithm(dir_path,nbits,iterations)


if __name__ == "__main__":

    main()