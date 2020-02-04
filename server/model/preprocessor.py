import pickle
import numpy as np
import os.path
import base64
from PIL import Image
import sys

class Preprocessor:
	def __init__(self):
		self.target_shape = [28,28]
		self.splitchar = "&"

	def preprocess(self, jpgtxt):
		# data = base64.decodestring(data)
		data = jpgtxt.split(',')[-1]
		data = base64.b64decode(data.encode('ascii'))

		g = open("temp.jpg", "wb")
		g.write(data)
		g.close()

		pic = Image.open("temp.jpg")
		pic = pic.resize(self.target_shape)
		M = np.array(pic) #now we have image data in numpy
		M = Preprocessor.rgb2gray(M)
		M = Preprocessor.normalize(M)
		M = M.reshape((1, 1, *self.target_shape))
		return M

	def dataset_length(self):
		with open(self.datafile) as f:
			for i, l in enumerate(f):
				pass
		return i + 1

	@staticmethod
	def squareTrim(M, min_side=20, threshold=0):
		assert M.shape[0]==M.shape[1],"Input matrix must be a square"
		wsum = np.sum(M,axis=0)
		nonzero = np.where(wsum > threshold*M.shape[1])[0]
		if len(nonzero) >= 1:
			wstart = nonzero[0]
			wend = nonzero[-1]
		else:
			wstart = 0
			wend = 0

		hsum = np.sum(M, axis=1)
		nonzero = np.where(hsum > threshold*M.shape[0])[0]
		if len(nonzero) >= 1:
			hstart = nonzero[0]
			hend = nonzero[-1]
		else:
			hstart=0
			hend = 0

		diff = abs((wend - wstart) - (hend - hstart))
		if (wend - wstart > hend - hstart):
			side = max(wend-wstart+1, min_side)
			m = np.zeros((side, side))
			cropped = M[hstart:hend+1,wstart:wend+1]
			shift = diff//2
			m[shift:cropped.shape[0]+shift, :cropped.shape[1]] = cropped
		else:
			side = max(hend-hstart + 1, min_side)
			m = np.zeros((side, side))
			cropped = M[hstart:hend + 1,wstart:wend + 1]
			shift = diff//2
			print('shift', shift)
			m[:cropped.shape[0], shift:cropped.shape[1]+shift] = cropped
		return m

	@staticmethod
	def rgb2gray(rgb):
		r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
		gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
		return gray

	@staticmethod
	def naiveInterp2D(M, newx, newy):
		result = np.zeros((newx, newy))
		for i in range(M.shape[0]):
			for j in range(M.shape[1]):
				indx = i*newx // M.shape[0]
				indy = j*newy // M.shape[1]
				result[indx, indy] += M[i, j]
		return result

	@staticmethod
	def normalize(M):
		max_val = np.max(M)
		return M / max_val
