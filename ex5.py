import cv2
import os,sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use( 'tkagg' )

def getHistVideo(file_path):

	cap = cv2.VideoCapture(file_path)
	#cap=cv2.VideoCapture(0)
	
	# Check if camera opened successfully 
	if (cap.isOpened()== False):  
		print("Error opening video file") 

	bins=256
	f1,ax=plt.subplots()
	ax.set_title("Histogram (RGB)")
	ax.set_xlabel("Bins")
	ax.set_ylabel("Frequency")
	lineR, = ax.plot(np.arange(bins), np.zeros((bins,)), c='r',label='Red')
	lineG, = ax.plot(np.arange(bins), np.zeros((bins,)), c='g',label='Green')
	lineB, = ax.plot(np.arange(bins), np.zeros((bins,)), c='b',label='Blue')
	ax.set_xlim(0,bins-1)
	
	ax.legend()



	f2,ax2=plt.subplots()
	ax2.set_title("Grayscale Histogram")
	lineGray, = ax2.plot(np.arange(bins), np.zeros((bins,1)), c='k', label='intensity')
	ax2.legend()
	ax2.set_xlim(0,bins-1)
	plt.ion()
	plt.show(block=False)

	
	while (1):
		# Capture frame-by-frame 
		ret, frame = cap.read()
		
		if ret == True:

			numPixels = np.prod(frame.shape[:2])
			ax.set_ylim(0,numPixels)
			ax2.set_ylim(0,numPixels)
			
			height , width , layers = frame.shape
			new_h=height*3
			new_w=width*3
			frame= cv2.resize(frame, (new_w, new_h)) 
			cv2.imshow('RGB', frame)

			(b,g,r)=cv2.split(frame)
			histogramR = cv2.calcHist([r], [0], None, [bins], [0, 255])
			histogramG = cv2.calcHist([g],[0],None,[bins],[0,255])
			histogramB = cv2.calcHist([b],[0],None,[bins],[0,255])

			lineR.set_ydata(histogramR)
			lineG.set_ydata(histogramG)
			lineB.set_ydata(histogramB)

			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			cv2.imshow("Gray",gray)
			histogram = cv2.calcHist([gray], [0], None, [bins], [0, 255])
			lineGray.set_ydata(histogram)

			f1.canvas.draw()
			f2.canvas.draw()
			# Press Q on keyboard to  exit 
			if cv2.waitKey(1) & 0xFF == ord('q'): 
				break

		else:
			print("Video ended")
			break

	cap.release()
	cv2.destroyAllWindows()


def getHistImage(file_path):

	img = cv2.imread(file_path,1)
	bins=256
	numPixels = np.prod(img.shape[:2])
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	fig,ax=plt.subplots(1,2)

	ax[0].set_title("RGB")
	ax[0].set_xlabel("Bins")
	ax[0].set_ylabel("Frequency")
	ax[0].set_xlim(0,bins-1)
	#ax[0].set_ylim(0,numPixels)

	ax[1].set_title("GrayScale")
	ax[1].set_xlim(0,bins-1)
	#ax[1].set_ylim(0,numPixels)

	lineR, = ax[0].plot(np.arange(bins), np.zeros((bins,)), c='r',label='Red') 
	lineG, = ax[0].plot(np.arange(bins), np.zeros((bins,)), c='g',label='Green')
	lineB, = ax[0].plot(np.arange(bins), np.zeros((bins,)), c='b',label='Blue')
	lineGray, = ax[1].plot(np.arange(bins), np.zeros((bins,1)), c='k', label='intensity')

	(b,g,r)=cv2.split(img)
	histogramR = cv2.calcHist([r], [0], None, [bins], [0, 255]) 
	histogramG = cv2.calcHist([g],[0],None,[bins],[0,255]) 
	histogramB = cv2.calcHist([b],[0],None,[bins],[0,255]) 
	histogram = cv2.calcHist([gray], [0], None, [bins], [0, 255])	

	lineR.set_ydata(histogramR)
	lineG.set_ydata(histogramG)
	lineB.set_ydata(histogramB)
	lineGray.set_ydata(histogram)

	maxR = max(histogramR)
	maxG = max(histogramG)
	maxB = max(histogramB)
	maxGr = max(histogram)

	x = max(*[ maxR, maxG, maxB]) 
	print(x)
	ax[0].set_ylim(0,x+5000)

	ax[1].set_ylim(0,maxGr+5000)

	plt.suptitle('Histogram', y=1.05, size=16)
	fig.tight_layout()
	plt.ion()
	plt.show(block=False)

	while(1):
		cv2.imshow("Image",img)
		cv2.imshow("Gray",gray)
		fig.canvas.draw()

		if cv2.waitKey(1) & 0xFF == ord('q'): 
			break



def main():

	mode = sys.argv[1]
	#mode = i/v (image or video)


	#while mode!= 'i' or mode!= 'v':
	#	print("Press i or v")
	#	option = int(input("Choose again: "))

	if mode == 'v':
		filenames= os.listdir ("./videos") # get all files' and folders' names in the current directory
		print("Videos: ", filenames)
		open_vid = str(input("Choose the video: "))

		for filename in filenames: # loop through all the files and folders

		    if open_vid == filename:
		        dir_path = str(os.path.join(os.path.abspath("./videos"),filename))
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

		getHistImage(dir_path)



if __name__ == "__main__":

    main()