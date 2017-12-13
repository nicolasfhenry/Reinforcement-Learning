# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 20:29:39 2017

@author: nicolas
"""

import os
import zipfile


print( '*' * 50)
print( '*' * 50)

print('1/7')
os.system('python mainGS_mirror_noVBN.py')

print( '*' * 50)
print( '*' * 50)

print('2/7')
os.system('python mainGS_mirror_VBN.py')


print( '*' * 50)
print( '*' * 50)

print('3/7')

os.system('python mainGS_noMirror_VBN.py')

print('4/7')

print( '*' * 50)
print( '*' * 50)

os.system('python mainGS_noVBN.py')

print('5/7')

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
zipf = zipfile.ZipFile('Results.zip', 'w', zipfile.ZIP_DEFLATED)
zipf.write('mainGS_mirror_noVBN.py.pkl') #1
zipf.write('mainGS_mirror_VBN.py.pkl') #2
zipf.write('mainGS_noMirror_VBN.py.pkl') #3
zipf.write('mainGS_noVBN.py.pkl') #4
zipf.write('mainGS_noVBN_noFitness.py.pkl') #5
zipf.write('main_noGS_noVBN.py.pkl') #6
zipf.write('main_noGS_VBN.py.pkl') #7

zipf.close()


zipf = zipfile.ZipFile('Params.zip', 'w', zipfile.ZIP_DEFLATED)
zipf.write('params-mainGS_mirror_noVBN.py.pkl') #1
zipf.write('params-mainGS_mirror_VBN.py.pkl') #2
zipf.write('params-mainGS_noMirror_VBN.py.pkl') #3
zipf.write('params-mainGS_noVBN.py.pkl') #4
zipf.write('params-mainGS_noVBN_noFitness.py.pkl') #5
zipf.write('params-main_noGS_noVBN.py.pkl') #6
zipf.write('params-main_noGS_VBN.py.pkl') #7



