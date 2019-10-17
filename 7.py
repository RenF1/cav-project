import wave, struct
from scipy.stats import norm
import os,sys
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import scipy.misc
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


def lloydAlgorithmImage(image_path,nbits,iter,epsilon=0.0001):

	img = cv2.imread(image_path,1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	
	(b,g,r)=cv2.split(img)
	q=np.power(2,nbits)
	step = 255/q

	ykGr = randomInit(gray,nbits)

	ykR= randomInit(r,nbits)
	ykG = randomInit(g,nbits)
	ykB = randomInit(b,nbits)

	epsR=epsGr=epsG=epsB=epsilon

	for i in range(0,iter):

		bkGr = getBorders(ykGr)

		bkR = getBorders(ykR)
		bkG = getBorders(ykG)
		bkB = getBorders(ykB)


		ykGr = reconstructValues(bkGr,gray,step)
		ykR = reconstructValues(bkR,r,step)
		ykG = reconstructValues(bkG,g,step)
		ykB = reconstructValues(bkB,b,step)


		xqGr = quantize(gray,bkGr,ykGr)

		xqR = quantize(r,bkR,ykR)
		xqG = quantize(g,bkG,ykG)
		xqB = quantize(b,bkB,ykB)

		epsGr = getError(gray,xqGr)

		epsR = getError(r,xqR)
		epsG = getError(g,xqG)
		epsB = getError(b,xqB)

		print("ErrR: ",epsGr)

		if epsilon>epsGr:
			minbk = bkGr
			minyk = bkGr
			break
		print("---------------------------\n")
		print("iter: ",i)



	xqGr = xqGr.astype(dtype='uint8')

	xqR = xqR.astype(dtype='uint8')
	xqG = xqG.astype(dtype='uint8')
	xqB = xqB.astype(dtype='uint8')


	image = cv2.merge((xqB,xqG,xqR))
	path1 = os.getcwd()+'/output/rgb'+str(nbits)+'b.png'
	path2 = os.getcwd()+'/output/gray'+str(nbits)+'b.png'
	cv2.imwrite(path1,image)
	cv2.imwrite(path2,xqGr)
	while(1):

		cv2.imshow("Quantized Gray Lloyd",xqGr)
		cv2.imshow("Original Gray ", gray)

		cv2.imshow("Original RGB",img)
		cv2.imshow("Quantized RGB Lloyd", image)



		if cv2.waitKey(1) & 0xFF == ord('q'): 
			break


def lloydAlgorithmVideo(file_path,nbits,iter,epsilon=0.0001):

	cap = cv2.VideoCapture(file_path)

	q=np.power(2,nbits)
	step = 255/q

	# Check if camera opened successfully 
	if (cap.isOpened()== False):  
		print("Error opening video file") 

	flag=True
	while (1):
		# Capture frame-by-frame 
		ret, frame = cap.read()
		
		if ret == True:

			height , width , layers = frame.shape
			new_h=height*3
			new_w=width*3
			frame= cv2.resize(frame, (new_w, new_h))

			if flag == True:
				size = (int(frame.shape[1]), int(frame.shape[0]))				
				path1 = os.getcwd()+'/output/rgb_video'+str(nbits)+'b.avi'
				path2 = os.getcwd()+'/output/gray'+str(nbits)+'b.avi'
				writerRGB = cv2.VideoWriter(path1, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, size, True)
				#writerGr = cv2.VideoWriter(path2, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, size, True)
				flag=False


			(b,g,r)=cv2.split(frame)
			
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			
			#ykGr = randomInit(gray,nbits)

			ykR= randomInit(r,nbits)
			ykG = randomInit(g,nbits)
			ykB = randomInit(b,nbits)

			for i in range(0,iter):

				#bkGr = getBorders(ykGr)

				bkR = getBorders(ykR)
				bkG = getBorders(ykG)
				bkB = getBorders(ykB)


				#ykGr = reconstructValues(bkGr,gray,step)
				ykR = reconstructValues(bkR,r,step)
				ykG = reconstructValues(bkG,g,step)
				ykB = reconstructValues(bkB,b,step)


				#xqGr = quantize(gray,bkGr,ykGr)

				xqR = quantize(r,bkR,ykR)
				xqG = quantize(g,bkG,ykG)
				xqB = quantize(b,bkB,ykB)

				#epsGr = getError(gray,xqGr)

				epsR = getError(r,xqR)
				epsG = getError(g,xqG)
				epsB = getError(b,xqB)

				if epsilon>epsR:
					minbkR = bkR
					minykR = ykR
				
				elif epsilon>epsG:
					minbkG = bkG
					minykG = ykG
				
				elif epsilon>epsB:
					minbkB = bkB
					minykB = ykB
				
				print("---------------------------\n")
				#print("iter: ",i) 

			#xqGr = xqGr.astype(dtype='uint8')

			xqR = xqR.astype(dtype='uint8')
			xqG = xqG.astype(dtype='uint8')
			xqB = xqB.astype(dtype='uint8')

			image = cv2.merge((xqB,xqG,xqR))
			#xqGr = cv2.cvtColor(xqGr, cv2.COLOR_GRAY2BGR)

			writerRGB.write(image)
			#writerGr.write(xqGr)
			
			# Press Q on keyboard to  exit 
			if cv2.waitKey(1) & 0xFF == ord('q'): 
				break

		else:
			print("Video ended")
			break

	cap.release()
	writerRGB.release()
	#writerGr.release()
	cv2.destroyAllWindows()

	
def randomInit(x,nbits):

	
	q=np.power(2,nbits)
	step = (np.max(x)-np.min(x))/q
	tmp = np.zeros_like(x)
	levels=np.linspace(np.min(x), np.max(x), num=q)
	middle_point = np.mean(x)

	var1 = np.arange(0,middle_point,step)
	
	var2 = np.arange(middle_point,255,step)
	
	if middle_point<127:
		var2 = np.delete(var2,-1)

	else:
		var1 = np.delete(var1,0)

	tmp = np.concatenate((var1,var2),axis=None)
	
	assert np.size(tmp) == q

	return tmp



def getBorders(x):

	tmp = np.zeros(np.size(x)-1)

	for i in range(len(x)-1):
		
		tmp[i] = 0.5*(x[i]+x[i+1])

	return tmp

def reconstructValues(x,channel,step):

	tmp= np.zeros(np.size(x)-1)

	for i in range(len(x)-1):
		if (np.abs(x[i+1]-x[i]) <=1 ):

			continue
		else:
			centroid = getExpected(x[i],x[i+1],channel)
			var = x[i+1]
			tmp[i] = centroid
	tmp = np.insert(tmp,0,0)
	tmp = np.append(tmp,var+step)
	tmp = np.sort(tmp,axis=None)
	
	return tmp


def quantize(x,b,y):
	
	tmp = np.zeros_like(x)
	w,h=np.shape(x)
	for i in range(len(b)-1):
		logical = np.logical_and(x>b[i],x <= b[i+1])
		
		comp = np.full(np.shape(tmp),y[i])

		tmp = np.where(logical,comp,tmp)

	return tmp


def getError(x,xq):

	assert np.size(x) == np.size(xq)

	return np.sum(np.power(x-xq,2))/np.size(x)

def getExpected(b1,b2,channel):

	expected = np.array([])
	den = np.array([])

	prob,vals=probability(channel)

	idx = np.rint(b1)
	idx2 = np.rint(b2)

	mask = ~np.any([(idx >= vals), (idx2 <= vals)], axis=0)

	for i in range (len(prob)-1):
		if mask[i]==True:
			expected=np.append(expected,prob[i]*vals[i])
			den = np.append(den,prob[i])

	expected=sum(expected)
	den = sum (den)

	if den is not 0:
		centroid = expected/den
	else:
		return (idx2-idx)*0.5
	
	return centroid


def probability(samples):

	numPixels = np.prod(samples.shape[:2])
	val, cnts = np.unique(samples, return_counts = True)
	prob = cnts/numPixels
	return prob,val		


def main():


	mode = sys.argv[1]
	nbits=int(sys.argv[2])
	iterations=int(sys.argv[3])
	
	if mode == 'v':
		filenames= os.listdir ("./videos") # get all files' and folders' names in the current directory
		print("Videos: ", filenames)
		open_vid = str(input("Choose the video: "))

		for filename in filenames: # loop through all the files and folders

		    if open_vid == filename:
		        dir_path = str(os.path.join(os.path.abspath("./videos"),filename))
	        	break

		lloydAlgorithmVideo(dir_path,nbits,iterations)

	elif mode == 'i':

		filenames= os.listdir ("./img_dataset") # get all files' and folders' names in the current directory
		print("Filenames: ", filenames)
		open_img = str(input("Choose the image: "))

		for filename in filenames: # loop through all the files and folders

		    if open_img == filename:
		        dir_path = str(os.path.join(os.path.abspath("./img_dataset"),filename))
	        	break

		plotPDF(dir_path)
		lloydAlgorithmImage(dir_path,nbits,iterations)


if __name__ == "__main__":

    main()