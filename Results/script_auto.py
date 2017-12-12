# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 20:29:39 2017

@author: nicolas
"""

import os
import zipfile



os.system('python mainGS_mirror_noVBN.py')
os.system('python mainGS_mirror_VBN.py')

os.system('python mainGS_noMirror_VBN.py')
os.system('python mainGS_noVBN.py')
os.system('python mainGS_noVBN_noFitness.py')

#ZipFile:
zipf = zipfile.ZipFile('Download.zip', 'w', zipfile.ZIP_DEFLATED)
ziph.write('mainGS_mirror_noVBN.py.pkl')
ziph.write('mainGS_mirror_VBN.py.pkl')
ziph.write('mainGS_noMirror_VBN.py.pkl')
ziph.write('mainGS_noVBN.py.pkl')
ziph.write('mainGS_noVBN_noFitness.py.pkl')
zipf.close()




