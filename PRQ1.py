# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import sklearn #Yao


#%% dispaly images

faces_mat = sio.loadmat('face.mat')
faces = faces_mat['X']


#for i in range(0, 520):
#    face = faces[:,i]
#    plt.subplot(52,10,i+1)
#    plt.imshow(np.reshape(face, (46,56)).T, cmap = 'gist_gray')
#    #plt.title('Face %d' % (i))
#    plt.xticks([]), plt.yticks([])
    
#%% data partition
count = 0
training_faces = np.zeros((2576, 1))
testing_faces = np.zeros((2576, 1))

#%%
for j in range(0,520):
    if count <8:
        training_faces = np.c_[training_faces, faces[:, j]]
        count += 1
    elif count == 8:
        testing_faces = np.c_[testing_faces, faces[:, j]]
        count += 1
    elif count == 9:
        testing_faces = np.c_[testing_faces, faces[:, j]]
        count = 0
        
testing_faces = testing_faces[:,1:]
training_faces = training_faces[:,1:]