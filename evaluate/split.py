import os
import cv2
from matplotlib import pyplot as plt
import uuid
import sys
import subprocess

def main(argv):

	if len(argv) == 0:
		raise Exception("No files given")

	files = [os.path.abspath(p) for p in argv if os.path.splitext(p)[1] == '.jpg']
	#files = files[-1:]

	nextParams = []
	for f in files:
		image = cv2.imread(f)
		
		imageGris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			
		blur = cv2.GaussianBlur(imageGris, (11,11), 0)
		(ret1,imageGris) = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
		
		image2,contours,hierarchy = cv2.findContours(imageGris.copy(), cv2.RETR_LIST , cv2.CHAIN_APPROX_SIMPLE)
			
		cnts = []
		firstPointsList = []
		for c in contours:
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.04 * peri, True)
			if len(approx) == 4:
				positions = cv2.boundingRect(approx)
				(x, y, w, h) = positions
				ar = w / float(h)
				shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
				if shape == "rectangle" and (y,y+h) not in firstPointsList:
					firstPointsList.append((y,y+h))
					cnts.append((c,peri, positions))
		
		newList = []
		btw=0
		NB_ELEMENTS = 4
		SPACE_BTW = 17
		sList = sorted(cnts, key=lambda c: c[1])
		sList.reverse()
		for i,t in enumerate(sList):
			if i !=0 and abs(sList[i-1][1] - t[1]) < SPACE_BTW:
				btw+=1
				if btw == NB_ELEMENTS and t[1] > 15:
					newList.extend(sList[i-NB_ELEMENTS-1:i])
					break
			else:
				btw=0
			
		#cv2.drawContours(image, contours, -1, (250, 0, 0), 10)
		#cv2.drawContours(image, [t[0] for t in cnts], -1, (0, 0, 250), 20)
		cv2.drawContours(image, [t[0] for t in newList], -1, (0, 250, 0), 30)
		
		imrvb = image[:,:,::-1]
		#plt.figure(figsize=(5,5))
		plt.imshow(imrvb)
		plt.title("Contours")
		#plt.show()
		
		
		newList = sorted(newList, key=lambda c: c[2][0])
		filesNames = []
		
		if len(newList) == 0:
			print(f)
			for _ in range(5):
				filesNames.append("NULL")
		else:
			for pos in newList:
				(x, y, w, h) = pos[2]
				img = image[y:y+h,x:x+w].copy()
				fileName = os.path.dirname(os.path.abspath(f))+'/sub_'+str(uuid.uuid4())+'.png'
				filesNames.append(fileName)
				cv2.imwrite(fileName, img)

		nextParams.extend(filesNames)
		
	args = 'python '+os.path.dirname(os.path.abspath(sys.argv[0]))+'/train.py '+' '.join(nextParams)
	#print(nextParams)
	cmd = subprocess.Popen(args, cwd=os.path.dirname(os.path.abspath(sys.argv[0])))
	print(cmd.communicate()[0])
if __name__ == "__main__":
	main(sys.argv[1:])