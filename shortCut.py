from __future__ import print_function, division
from TAPS import *
from Confs import Confs
import time
import mdtraj as md
import numpy as np
import shutil

dirPars = 'pars'            # dirPar
parFile = 'taps.par'        # parameters filename
topFile = 'protein.pdb'     # topology filename
p0File = 'path0.xtc'   # initial path file
alignFile = 'align.ndx'      # atom index file for alignment
rmsFile =  'rms.ndx'         # atom index file for rmsd computation

taps = TAPS(dirPars, parFile, topFile, p0File, alignFile, rmsFile)	#Finished#
p0 = taps.refPath
p1 = p0.rmClose(0.12)
p1.nodes.save('p1_rc012.xtc')
p1 = p0.rmClose(0.10)
p1.nodes.save('p1_rc010.xtc')


