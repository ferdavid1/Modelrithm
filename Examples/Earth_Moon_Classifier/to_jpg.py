from PIL import Image
import os
filelist = []
os.chdir('images/mars')
for c in os.listdir():
	filelist.append(c)
print(filelist)
#print(os.getcwd())
for infile in filelist:
	outfile= os.path.splitext(infile)[0] + ".jpg" # outfile = file in path jpg ending
	if infile != outfile: # if each file isnt one of the existing jpg files...
		try:
			Image.open(infile).save(outfile)
		except IOError:
			print("cannot convert", infile)