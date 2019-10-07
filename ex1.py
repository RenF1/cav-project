import wave, struct
from pydub import AudioSegment
from pydub.playback import play
import os,sys

def copy(file_to_open,file_to_write):

	waveFile = wave.open(file_to_open, 'rb')
	wavWrite=wave.open(file_to_write,'wb')


	nchannels, sampwidth, framerate, nframes, comptype, compname = waveFile.getparams()
	wavWrite.setparams((nchannels, sampwidth, framerate, nframes,comptype, compname))


	print("nChannels: ",nchannels)
	print("Sampwidth: ", sampwidth)
	print("Framerate: ", framerate)
	print("nFrames: ", nframes)
	print("CompType: ", comptype)
	print("CompName: ", compname)

	fmt = "<" + 'h' * nchannels	

	while waveFile.tell() < nframes:
		decoded = struct.unpack(fmt, waveFile.readframes(1))
		out = struct.pack(fmt,decoded[0],decoded[1])
		wavWrite.writeframes(out)

	waveFile.close()
	wavWrite.close()

def main():


	open_file = sys.argv[1]
	write_file=sys.argv[2] 

	filenames= os.listdir ("./WAVfiles") # get all files' and folders' names in the current directory
	print("Filenames: ", filenames)

	for filename in filenames: # loop through all the files and folders

	    if open_file == filename:
	        dir_path = str(os.path.join(os.path.abspath("./WAVfiles"),filename))
        	break

	print("Ficheiro a ser lido: ", dir_path)
	copy(dir_path,write_file)

	sound = AudioSegment.from_file(write_file, format="wav")
	play(sound)

if __name__ == "__main__":

    main()