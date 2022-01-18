import math
import numpy as np

class Evaluation:

	def compute_rank1(self, Y, y):
		# for i in range(0,len(Y)):
		# 	Y[i][i] = math.inf

		classes = np.unique(sorted(y))
	
		count_all = 0
		count_correct = 0
		for cla1 in classes:
			idx1 = y==cla1
	
			if (list(idx1).count(True)) <= 1:
				continue
			# Compute only for cases where there is more than one sample:
			Y1 = Y[idx1==True, :]
			
			Y1[Y1==0] = math.inf
			for y1 in Y1:
				s = np.argsort(y1)
				smin = s[0]
				imin = idx1[smin]
				count_all += 1
				if imin:
					count_correct += 1
		return count_correct/count_all*100


	# Add your own metrics here, such as rank5, (all ranks), CMC plot, ROC, ...
'''
	def compute_rank5(self, Y, y):
	# First loop over classes in order to select the closest for each class.
		# for i in range(0,len(Y)):
		# 	Y[i][i] = math.inf

		classes = np.unique(sorted(y))
		
		sentinel = 0
		for cla1 in classes:
			idx1 = y==cla1
			if (list(idx1).count(True)) <= 1:
				continue
			Y1 = Y[idx1==True, :]
			print("CLA1")
			print(cla1)
			print(Y1)
			for cla2 in classes:
				# Select the closest that is higher than zero:
				idx2 = y==cla2
				if (list(idx2).count(True)) <= 1:
					continue
				Y2 = Y1[:, idx1==True]
				print("CLA2")
				print(cla2)
				print(Y2)
				Y2[Y2==0] = math.inf
				min_val = np.min(np.array(Y2))
				# ...

		
	def cmc_curve(negatives, positives):
		bob.measure.cmc(rank)

	def roc_curve(npoint, negatives, positives):
		from matplotlib import pyplot
		npoints = 100
		bob.measure.plot.roc(negatives, positives, npoints, color=(0,0,0), linestyle='-', label='test') 
		pyplot.xlabel('FAR (%)') 
		pyplot.ylabel('FRR (%)') 
		pyplot.grid(True)
		pyplot.show() '''