__author__ = 'Lizhe Zhu'

import numpy as np

class PcvInd(object):
	"""
	attributes:
	atomInd     numpy.int:      combined set of input align & input rms:  (atomIndex in PDB, atomInd-1=index in TAPS)
	align       numpy.int:      which atoms in atomInd to align trajectory: for MDTraj
	rms         numpy.int:      which atoms in atomInd to compute RMSD : for MDTraj
	alignPLU    numpy.float:    output numpy array, if used to align:  for Plumed
	rmsPLU      numpy.float:    output numpy array, if used to compute rmsd: for Plumed
	"""
	def __init__(self, align = None, rms = None):		#Finished#
		"""
		:param align: index for alignment, numpy array (1,x)
		:param rms: index for RMSD computation numpy array (1,x)
		"""
		if (align is None) or (rms is None):
			self.atomInd, self.align, self.rms, self.atomSlice, self.alignPLU, self.rmsPLU = None, None, None, None, None, None
		else:
			a=align.astype(int)
			r=rms.astype(int)
			self.atomInd = np.unique(np.append(a, r))	#合在一起，并去掉重复的，再按由小到大重新排序#
			self.atomSlice = np.subtract(self.atomInd,1)	#给每项都减1#
			self.align, self.rms = np.zeros(len(align),dtype=np.int), np.zeros(len(rms),dtype=np.int)
			tmp = len(self.atomInd)
			self.alignPLU, self.rmsPLU = np.zeros(tmp), np.zeros(tmp)
			for ia in range(len(align)):
				id = np.where(self.atomInd == align[ia])[0]		#输出align和rms合在一起后，等于原来align[ia]的所有位置（编号），[0]表示第一个#
				self.align[ia] = id 	#跟新的tmp序列保持一致#
				self.alignPLU[id] = 1.00	#1.00 表示把tmp（即atomInd中所有原子编号）中将会用来align的原子#
			for ir in range(len(rms)):	#这部分跟上面align类似#
				id = np.where(self.atomInd == rms[ir])[0]
				self.rms[ir] = id
				self.rmsPLU[id] = 1.00	#1.00 表示会作为rms计算的原子#

