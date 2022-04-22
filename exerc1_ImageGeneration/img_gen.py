""" 
Name: Lucas Henrique Sant'Anna 
USP number: 10748521
SCC0251 - Digital Image Processing
Assigment 1 - Image Generation
2022.1 
"""

import numpy as np
from random import randint, random, seed

#Function Normalize
def normalize(mtx, num):# Normalize the matrix values between 0 and num
    mtx_norm = (mtx - np.min(mtx)) * num / np.max(mtx)
    
    return mtx_norm
#----------------------------------------------

#Function Downsampling to new_size x new_size
def downsampling(mtx_orig, new_size):
    new_mtx = np.zeros((new_size,new_size), dtype = float) 
    ratio = len(mtx_orig) // new_size #defines the ratio of the step in mtx_orig
    
    for x in range(new_size):
        for y in range(new_size):
            new_mtx[x, y] = mtx_orig[ratio * x, ratio * y]  
    
    return new_mtx
#----------------------------------------------

#Function Quantisation matrix values to B bits
def quant8bits(mtx, bits):
    mtx = normalize(mtx, 255).astype(np.uint8) #normalize between 0 and 8 
    quant_mtx = mtx >> 8 - bits #bitwise right shift
    
    return quant_mtx
#-----------------------------------------------

#function Root Square Error 
def rse(mtx1, mtx2):
    error = np.sum((mtx1 - mtx2)**2)
    
    return error**(1/2)
#-----------------------------------------------

filename = str(input()).rstrip() 
R = np.load(filename) # Reference Image R / a numpy file (.npy)

C= int(input()) # Size of Scene Image (square C x C)
func_slct= int(input()) # Select the function that will be use to create Scene Image]
Q = int(input()) # Parameter Q

#Size of the DIgital Image N (square N x N)
N = int(input())
if N > C : quit() # N must be lower than C 

#Number of bits per pixel B
B = int(input())
if B < 1 or B > 8 : quit() # 1 <= B <= 8

S = int(input()) # seed for the random fuction

#Generate Scene Image -----
# Options to generate
img_f = np.zeros((C,C), dtype = float) #create a matrix of zeros
if func_slct == 1:
    for x in range(C):
        for y in range(C):
            img_f[x, y] = x*y + 2*y

elif func_slct == 2:
    for x in range(C):
        for y in range(C):
            img_f[x, y] = abs(np.cos(x/Q) + 2*np.sin(y/Q))

elif func_slct == 3:
    for x in range(C):
        for y in range(C):
            img_f[x, y] = abs(3*(x/Q) - (y/Q)**(1/3)) 
            
elif func_slct == 4: # random function
    random.seed(S) #seed S
    for x in range(C):
        for y in range(C):
            img_f[x, y] = random()

elif func_slct == 5: # random walk
    seed(S) #seed S
    x = 0
    y = 0
    img_f[x, y] = 1 # Define img_f[0,0] = 1
    for i in range((C*C)+ 1):
        dx = randint(-1, 1) # integer range [-1, 1] / from-1 to 1
        dy = randint(-1, 1)
        x = np.mod((x + dx), C) #walk in x direction
        y = np.mod((y + dy), C) #walk in y direction
        img_f[x, y] = 1 #define current value 

#Normalize between 0 and 65355         
img_f = normalize(img_f, 2**16 - 1) # normaliza f

#downsampling image from CxC to NxN
img_g = downsampling(img_f, N)

#quantisation of matrix values
img_g = quant8bits(img_g, B)

#RSE
_rse = rse(img_g, R)
print(np.round(_rse, 4))  #print with the rounded 4 decimal palaces