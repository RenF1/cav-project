import os,sys
import cv2
import numpy as np

def copy(img_to_open,img_to_write):
	img = cv2.imread(img_to_open,1)
	(B,G,R) = cv2.split(img)
	rows = img.shape[0]
	cols = img.shape[1]	
	channels = img.shape[2] 
	print("Height: ", rows) 
	print("Width: ", cols)
	print("Channels: ", channels)
	new= np.zeros((rows,cols,3), np.uint8)
	for i in range (rows):
		for j in range (cols):
			new[i,j] = img[i,j]

	return (img,new)

def main():

	open_img = sys.argv[1]
	write_img = sys.argv[2] 

	filenames= os.listdir ("./img_dataset") # get all files' and folders' names in the current directory
	print("Filenames: ", filenames)

	for filename in filenames: # loop through all the files and folders

	    if open_img== filename:
	        dir_path = str(os.path.join(os.path.abspath("./img_dataset"),filename))
        	break

	print("Ficheiro a ser lido: ", dir_path)
	x = copy(dir_path,write_img)
	print (len(x))
	oimg=x[0]
	wimg=x[1]
	cv2.imwrite(os.path.join(os.path.abspath("./img_dataset"),write_img),wimg)

	while (1):
		key=cv2.waitKey(1)
		if key == ord ('q'): break
		cv2.imshow("Original image", oimg)
		cv2.imshow("Image Copied", wimg)


if __name__ == "__main__":

    main()