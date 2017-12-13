# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 20:29:39 2017

@author: nicolas
"""

import os
import zipfile


print( '*' * 50)
print( '*' * 50)

print('1/5')
os.system('python mainGS_mirror_noVBN.py')

print( '*' * 50)
print( '*' * 50)

print('2/5')
os.system('python mainGS_mirror_VBN.py')


print( '*' * 50)
print( '*' * 50)

print('3/5')

os.system('python mainGS_noMirror_VBN.py')

print('4/5')

print( '*' * 50)
print( '*' * 50)

os.system('python mainGS_noVBN.py')

print('5/5')

print( '*' * 50)
print( '*' * 50)
os.system('python mainGS_noVBN_noFitness.py')

print( '*' * 50)
print( '*' * 50)

print('6/7')

print( '*' * 50)
print( '*' * 50)
os.system('python main_noGS_noVBN.py')

print( '*' * 50)
print( '*' * 50)

print('6/7')

os.system('python main_noGS_VBN.py')

print( '*' * 50)
print( '*' * 50)

#ZipFile:
zipf = zipfile.ZipFile('Download.zip', 'w', zipfile.ZIP_DEFLATED)
zipf.write('mainGS_mirror_noVBN.py.pkl')
zipf.write('mainGS_mirror_VBN.py.pkl')
zipf.write('mainGS_noMirror_VBN.py.pkl')
zipf.write('mainGS_noVBN.py.pkl')
zipf.write('mainGS_noVBN_noFitness.py.pkl')
zipf.write('main_noGS_noVBN.py.pkl')
zipf.write('main_noGS_VBN.py.pkl')

zipf.close()




