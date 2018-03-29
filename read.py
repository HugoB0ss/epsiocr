from PIL import Image
im = Image.open('images/img.png')
pix = im.load()
x_im, y_im = im.size
newImg = []
for y in range(0,y_im):
	vals = []
	for x in range(0, x_im):
		vals.append(pix[x,y][0])
	newImg.append(vals)
print(newImg)