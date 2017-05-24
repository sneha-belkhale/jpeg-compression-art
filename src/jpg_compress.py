import numpy as np
import cv2
import math
import sys


def compress(image_dct, image_orig, start_pos_x, start_pos_y, n):
	"""
	transform an nxn image block from pixel intensity to frequency domain using the discrete cosine transform.

	@param DCT : resulting image in the frequency domain
	@param image_orig: original image in the pixel intensity domain
	@param start_pos_x: row value of top corner of block
	@param start_pos_y: column value of top corner of block
	@param n: size of square block

	"""
	for i in range (0,n):
		for j in range (0,n):
			coef_sum = 0
			cI = 1
			cJ = 1
			for x in range (0,n):
				for y in range (0,n):
					#coef_sum += (image_orig[start_pos_x + x, start_pos_y + y]*math.cos(((2*x+1)*i*math.pi)/(2*n))*math.cos(((2*y+1)*j*math.pi)/(2*n)))
					coef_sum += (image_orig[start_pos_x + x, start_pos_y + y]*coef[i,j,x,y])
			if (i==0):
				cI = 1/math.sqrt(2)
			if (j==0):
				cJ = 1/math.sqrt(2)
			image_dct[start_pos_x + i,start_pos_y + j] = cI*cJ*coef_sum/math.sqrt(2*n)


def quantize(image_quantized, image_dct, quant, start_pos_x, start_pos_y, n):
	"""
	quantize an image block, given a specific quantization matrix. This is the step where we are losing information, so
	changing the quantization matrix and size of blocks (n) is going to give us control on the compression results.

	@param image_quantized: resulting image after dequantization
	@param image_dct : transformed image in the frequency domain
	@param quant: given quantization matrix, must be of size nxn
	@param start_pos_x: row value of top corner of block
	@param start_pos_y: column value of top corner of block
	@param n: size of square block
	"""
	x=0
	y=0
	for i in range (start_pos_x, start_pos_x + n):
		y=0
		for j in range(start_pos_y , start_pos_y + n):
			# TODO losing information!!!
			image_quantized[i,j] = math.floor((image_dct[i,j]+quant[x,y]/2)/quant[x,y])
			y = y+1
		x=x+1


def dequantize(image_dequantized, image_quantized, quant, start_pos_x, start_pos_y, n):
	"""
	dequantize an image block, given a specific quantization matrix

	@param image_dequantized: resulting image after dequantization
	@param image_quantized : quantized image in the frequency domain
	@param quant: given quantization matrix, must be of size nxn
	@param start_pos_x: row value of top corner of block
	@param start_pos_y: column value of top corner of block
	@param n: size of square block
	"""
	x=0
	y=0
	for i in range (start_pos_x, start_pos_x + n):
		y=0
		for j in range(start_pos_y , start_pos_y + n):
			image_dequantized[i,j] = image_quantized[i,j]*quant[x,y]
			y=y+1
		x=x+1


def decompress(image_reconstructed, image_dequantized, start_pos_x, start_pos_y, n):
	"""
	transform an nxn image block from frequency to pixel intensity domain using
	the inverse discrete cosine transform.

	@param image_reconstructed: resulting image in the pixel intensity domain
	@param image_dequantized : dequantized image in the frequency domain
	@param start_pos_x: row value of top corner of block
	@param start_pos_y: column value of top corner of block
	@param n: size of square block

	"""
	for x in range (0,n):
		for y in range (0,n):
			coef_sum = 0
			for u in range (0,n):
				for v in range (0,n):
					cU = 1
					cV = 1
					if (u==0):
						cU = 1/math.sqrt(2)
					if (v==0):
						cV = 1/math.sqrt(2)
					coef_sum += (cU*cV*image_dequantized[start_pos_x + u,start_pos_y + v]*coef_inv[x,y,u,v])
			image_reconstructed[start_pos_x + x,start_pos_y + y] = np.int(coef_sum/math.sqrt(2*n))

def createCoefficientMatrix(n):
	"""
	create a coefficient matrix for discrete cosine transform, for faster computation
	@param n: size of square block
	"""
	coef = np.zeros([n,n,n,n])
	for i in range (0,n):
		for j in range (0,n):
			for x in range (0,n):
				for y in range (0,n):
					coef[i,j,x,y] = math.cos(((2*x+1)*i*math.pi)/(2*n))*math.cos(((2*y+1)*j*math.pi)/(2*n))
	return coef

def createInvCoefficientMatrix(n):
	"""
	create a coefficient matrix for the inverse discrete cosine transform, for faster computation
	@param n: size of square block
	"""
	coef_inv = np.zeros([n,n,n,n])
	for x in range (0,n):
		for y in range (0,n):
			for u in range (0,n):
				for v in range (0,n):
					coef_inv[x,y,u,v] = math.cos(((2*x+1)*u*math.pi)/(2*n))*math.cos(((2*y+1)*v*math.pi)/(2*n))
	return coef_inv


"""
main script
"""


img_filename = sys.argv[1]
#set compression block size. this will effect the size of the grid pattern in the decompressed image
n = np.int8(sys.argv[2])
#chose value for a uniform quantization
quant_val = np.int8(sys.argv[3])

#read in the original image
img_original_rgb = cv2.imread(img_filename)

#we will focus on compressing the green channel for an interesting effect
img_original = img_original_rgb[:,:,1]
#img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

cv2.imshow("original", img_original_rgb)

#set up the placeholders
img_dct = np.zeros(img_original.shape)
img_quantized = np.zeros(img_original.shape)
img_dequantized = np.zeros(img_original.shape)
img_reconstructed = np.zeros(img_original.shape)

np.set_printoptions(threshold='nan')


"""
quantization matrix. this matrix must be the same size as the image block you want to transform. For example, analyzing this matrix shows that the values
gradually increase in magnitude moving from the top left to bottom right. This means that when dividing your image block by this quantization matrix,
it is more likely that the bottom right pixels are going to be closer to zero -- and lost in quantization.
this is a common pattern in quantization of the frequency domain because the frequencies not crucial to the human eye will be stored in the
bottom right corner and should be the first to be lost.

example quantization matrix:

quant = np.array((
 [16, 11, 10, 16, 24, 40, 51, 61],
 [12, 12, 14, 19, 26, 58, 60, 55],
 [14, 13, 16, 24, 40, 57, 69, 56],
 [14, 17, 22, 29, 51, 87, 80, 62],
 [18, 22, 37, 56, 68, 109, 103, 77],
 [24, 35, 55, 64, 81, 104, 113, 92],
 [49, 64, 78, 87, 103, 121, 120, 101],
 [72, 92, 95, 98, 112, 100, 103, 99]
))
"""
#here we chose a uniform quantization
quant = np.ones([n,n])*quant_val

#create a coefficient matrix for discrete cosine transform, for faster computation
coef = createCoefficientMatrix(n)
#create a coefficient matrix for inverse discrete cosine transform, for faster computation
coef_inv = createInvCoefficientMatrix(n)


#loop through the entire image in blocks. i and j represent the top left corner of the current image block being transformed.
for i in range(0, img_original.shape[0]/n):
	for j in range(0, img_original.shape[1]/n):
		#print "original: ",mtx[i*8:(i*8+8),j*8:(j*8+8)]
		compress(img_dct, img_original, i*n, j*n, n)
		#print "transformed", DCT[i*n:(i*n+n),j*n:(j*n+n)]
		quantize(img_quantized, img_dct, quant, i*n, j*n, n)
		#print "quantized", DCT[i*n:(i*n+n),j*n:(j*n+n)]
		dequantize(img_dequantized, img_quantized, quant, i*n, j*n, n)
		#print "dequantized", DCT[i*n:(i*n+n),j*n:(j*n+n)]
		decompress(img_reconstructed,img_dequantized, i*n,j*n, n)
		#print "decompressed", image_reconstructed[i*8:(i*8+8),j*8:(j*8+8)]

"""
opencv's imshow function has specific rules for displaying intensity values based on the data type of the matrix.
imshow expects intensity values between 0-1 for 32-bit data type images
										0-255 for 8-bit integer data type images

img_reconstructed is values from 0-255. However, since img_reconstructed is a 64-bit data type, we must map the intensities to 0-1 and convert to 32-bit in order to be
correctly understood by cv2.imshow function.
We can also simply convert the img_reconstructed to np.int8 and use values as is.
"""
#cv2.imshow("reconstructed", np.float32(img_reconstructed * 1.0/255)
#make sure that the values are between 0 and 255.
low_values_indices = img_reconstructed < 0
img_reconstructed[low_values_indices] = 0
high_values_indices = img_reconstructed > 255
img_reconstructed[high_values_indices] = 255

#adjust the green channel to the new decompressed values
img_original_rgb[:,:,1] = np.int8(img_reconstructed)

cv2.imshow("reconstructed", img_original_rgb)
while(1):
    if( cv2.waitKey(100) == 27 ):
		break

#cv2.waitKey(0)
