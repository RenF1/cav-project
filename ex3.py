import os,sys
import cv2
import numpy as np

def main():

	print(" 0 - Video")
	print(" 1 - Image")
	option = int(input("Enter the option: "))

	while option >= 2:
		print("Press 0 or 1")
		option = int(input("Choose again: "))

	if option == 0:
		filenames= os.listdir ("./videos") # get all files' and folders' names in the current directory
		print("Videos: ", filenames)
		open_vid = str(input("Choose the video: "))

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

				height , width , layers = frame.shape
				new_h=height*3
				new_w=width*3
				resize = cv2.resize(frame, (new_w, new_h)) 
			   
				# Display the resulting frame 
				cv2.imshow('Frame', resize) 

				# Press Q on keyboard to  exit 
				if cv2.waitKey(33) & 0xFF == ord('q'): 
					break
			   
			# Break the loop 
			else:  
				break

		cap.release()


	elif option == 1:

		filenames= os.listdir ("./img_dataset") # get all files' and folders' names in the current directory
		print("Filenames: ", filenames)
		open_img = str(input("Choose the image: "))

		for filename in filenames: # loop through all the files and folders

		    if open_img == filename:
		        dir_path = str(os.path.join(os.path.abspath("./img_dataset"),filename))
	        	break

		img = cv2.imread(dir_path,1)
    	
		while(1):

			key=cv2.waitKey(1) & 0xFF
			if key == ord ('q'): break
			cv2.imshow("Original image", img)

cv2.destroyAllWindows()

if __name__ == "__main__":

    main()

