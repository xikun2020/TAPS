from __future__ import print_function, division
# =======================================================================================================================
#                           import mpi4py for parallel computing
# =======================================================================================================================
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# ======================================================================================================================
#                                       digits formater for iterations: "3 --> 03"
# ======================================================================================================================
def digits(s1):
	s2 = "%.3d" % s1
	return s2

from TAPS import *
#from Confs import Confs
import time
import errno
import copy
import os
from copy import deepcopy
# =========================================================================================================
#                                         multiple independent taps
# =========================================================================================================
n_start = 0
n_taps = 1

# =========================================================================================================
#                                       number of iterations per taps
# =========================================================================================================
n_iter = 120
iter_start = 0

# =========================================================================================================
#                                                input files
# =========================================================================================================
dirPars = 'pars'            # dirPar
parFile = 'taps.par'        # parameters filename
topFile = 'protein.pdb'     # topology filename
p0File = 'path' + str(iter_start) + '.xtc'   # initial path file
alignFile = 'align.ndx'      # atom index file for alignment
rmsFile = 'rms.ndx'         # atom index file for rmsd computation
ndxFile = 'index.ndx'

for i in range(n_start,n_taps+n_start):
	tapsName = 'PathInd' + str(i)
	#print("+++DEBUG+++ Barrier before iteration")
	comm.Barrier()

	if rank == 0 and not os.path.exists(tapsName):
		try:
			os.makedirs(tapsName)
		except OSError as error:
			if error.errno != errno.EEXIST:
				raise #Here, just make directory for saving calculation data#
	else:
	    time.sleep(5)
	
	comm.Barrier()

	if rank == 0:

		print(tapsName, ":")
		print("+++TAPS+++   Reading input parameters") 

		t0 = time.time()
		#print("+++DEBUG+++ Size:", size, "Rank:", rank, "running TAPS")

		taps = TAPS(dirPars, parFile, topFile, p0File, alignFile, rmsFile, ndxFile)	#Create file system and read parameters#

		te = time.time()
		print("+++TAPS+++   Reading finished (time-cost:", te - t0, 'sec)')
		pathList = []

		refPath = copy.copy(taps.refPath)	#Initialize parameters and conformation for taps run#
		pathList.append(refPath)
		dirEvol = tapsName + '/paths'
		if not os.path.exists(dirEvol):
			os.makedirs(dirEvol)
		refPath.pathName = 'iter' + digits(iter_start)
		refPath.exportPath(dirEvol)
	else:
		taps = None
		refPath = None

	comm.Barrier()

	taps = comm.bcast(taps, root=0)
	refPath = comm.bcast(refPath, root=0)

	comm.Barrier()

	for j in range(iter_start, iter_start + n_iter): #Here, n_iter => times of iteration# 
		# ==================================================================================================
		#                                          iteration index
		# ==================================================================================================
		itr = 'iter' + digits(j+1)
		# ==================================================================================================
		#                                        one taps iteration
		# ==================================================================================================
		dirMeta = tapsName + '/sampling/' + itr		

		if rank == 0:
			print("+++TAPS+++  ", itr, ": Preparing perpendicular sampling")
			dirRUNs = taps.meta_dirs(refPath, dirMeta)	#Create working directories#
			ti = time.time()
			taps.meta_setup(refPath, dirMeta, dirRUNs)            
			te = time.time()
			print("+++TAPS+++  ", itr, ": Sampling preparation finished (time-cost: ", te - ti, 'sec)')
			print("+++TAPS+++  ", itr, ": Start perpendicular sampling")
		else:
			dirRUNs = None

		comm.Barrier()
		dirRUNs = comm.bcast(dirRUNs, root=0)
		comm.Barrier()
		taps.meta_sample(dirMeta, dirRUNs)
		comm.Barrier()
		
		if rank == 0:
			tf = time.time()
			print("+++TAPS+++  ", itr, ": Perpendicular sampling finished (time-cost: ", tf - te, 'sec)')
			print("+++TAPS+++  ", itr, ": Analyzing data to update path")
			
		comm.Barrier()
		p_meta = taps.meta_analyze(dirMeta, dirRUNs)
		comm.Barrier()

		if rank == 0:
			tg = time.time()
			print("+++TAPS+++  ", itr, ": Analysis finished (time-cost: ", tg - tf, 'sec)')
			p_meta.pathName = itr
			p_meta.exportPath(dirEvol)
			print(' ')
			refPath = deepcopy(p_meta)

		comm.Barrier()

	if rank== 0:
		print('Path Optimization finished.')
	comm.Barrier()

MPI.Finalize()  
