# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# git add .
# git commit -m "version 02"
# git push

#%%
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#%% import faces and labels

faces_mat = sio.loadmat('face.mat') #load faces from file
faces = faces_mat['X']
labels = faces_mat['l']


#%% data partition
count = 0
faces_train = np.zeros((2576, 1))
faces_test = np.zeros((2576, 1))

for i in range(520):
    if count <8:
        faces_train = np.c_[faces_train, faces[:, i]] #attach first 8 vector image of each class to faces_train
        count += 1
    elif count == 8:
        faces_test = np.c_[faces_test, faces[:, i]] #attach 9th vector image of each class to faces_test
        count += 1
    elif count == 9:
        faces_test = np.c_[faces_test, faces[:, i]] #attach 10th vector image of each class to faces_test
        count = 0
faces_test = faces_test[:,1:] #pop first zero vector
faces_train = faces_train[:,1:] #pop first zero vector


#%% Q1.a
average_face = faces_train.mean(1) #mean face
average_face_mat = np.tile(average_face, (416,1)).T #copy the mean face vector for each column in the matrix
A = faces_train - average_face_mat
covariance_mat = np.matmul(A, A.T)/416
eigvals, eigvecs = np.linalg.eigh(covariance_mat) #returns eigenvalues in order and normalized eigenvectors in the eigenvalues order
eigvals = np.flip(eigvals, 0) #flip eigenvalues vector
eigvecs = np.flip(eigvecs, 1) #flip eigenvectors matrix

#normalize eigenvectors
#eigvecs_norm = np.divide(eigvecs, np.linalg.norm(eigvecs, axis = 0))

#normalize eigenvalues
#eigvals_norm = eigvals / np.linalg.norm(eigvals)

plt.imshow(np.reshape(average_face, (46,56)).T, cmap = 'gist_gray') #print mean face
plt.show()

count = 0
for i in range (2576): #number of non-zero eigenvalues
    if eigvals[i] < 1:
        count += 1
print("number of non-zero eigenvalues: %s" % (2576-count))


plt.title('eigenvals values - log')
plt.semilogy(abs(eigvals[:])) #graph of eigenvalues values (log)
plt.grid(True)
plt.show()

plt.plot(abs(eigvals[:])) #graph of eigenvalues values (linear)
plt.grid(True)
plt.title('eigenvals values- linear')
plt.show()

#sum elementwise eigval, divide each element for the total sum (normalize) and see the influence of each eigevalue

#%% Q1.b
covariance_mat_ld = np.matmul(A.T, A)/416 #low dimensional covariance matrix
eigvals_ld, eigvecs_ld = np.linalg.eigh(covariance_mat_ld) #returns eigenvalues in order and normalized eigenvectors in the eigenvalues order

count = 0
for i in range(416): #number of non-zero eigenvalues
    if eigvals_ld[i] < 1:
        count += 1
print("number of non-zero eigenvalues: %s" % (416-count))

eigvals_ld = np.flip(eigvals_ld, 0) #flip eigenvalues_ld vector
eigvecs_ld = np.flip(eigvecs_ld, 1) #flip eigenvectors_ld matrix

plt.semilogy(abs(eigvals_ld[:])) #graph of eigenvalues_ld values (log)
plt.grid(True)
plt.title('eigenvals_ld values - log')
plt.show()
    
plt.plot(abs(eigvals_ld[:])) #graph of eigenvalues_ld values (linear)
plt.grid(True)
plt.title('eigenvals_ld values - linear')
plt.show()


#%% Q2.a

#declare matrices
omega = np.zeros((416,1))
eigvecs_new = np.zeros((2576,416))

#calculate u = Av and normalize u
for i in range(416):
    eigvecs_new[:,i] = np.matmul(A,eigvecs_ld[:,i])
    eigvecs_new[:,i] = eigvecs_new[:,i] / np.linalg.norm(eigvecs_new) #normalization
    
for i in range(416): #calculat omega
    omega[i,0] = np.matmul(A[:,3].T,eigvecs_new[:,i]) #change A column value to change image

#print original face
face = faces_train[:,3] #change A column value to change image #switch row with column with different partition alghoritm
plt.imshow(np.reshape(face, (46,56)).T, cmap = 'gist_gray')
plt.xticks([]), plt.yticks([])
plt.show()

sum = 0
for i in range(100): #face reconstruction
    to_add = omega[i,0]*eigvecs_new[:,i]
    sum += to_add
face_r = average_face + sum

#print reconstructed face
plt.imshow(np.reshape(face_r, (46,56)).T, cmap = 'gist_gray')
plt.xticks([]), plt.yticks([])
plt.show()

#%% Q2.b
face_new = faces_test[:, 3] #get an image from the test set
face_minus_average = face_new - average_face

plt.imshow(np.reshape(face_new, (46,56)).T, cmap = 'gist_gray')
plt.xticks([]), plt.yticks([])
plt.show()

omega_all = np.zeros((416,416))
for i in range(416): #calculate omega matrix with all the omegas of the training set
    for j in range(416):
        omega_all[j,i] = np.matmul(A[:,i].T,eigvecs_new[:,j]) 


omega_new = np.zeros((416,1))
for i in range(0,416): #calculate omega of test image
    omega_new[i,0] = np.matmul(face_minus_average.T,eigvecs_new[:,i])

get_the_min = omega_new - omega_all

min_distance = 0
for i in range(416):
    distance = np.linalg.norm(get_the_min[:,i])
    if distance <=  np.linalg.norm(get_the_min[:,min_distance]):
        min_distance = i
        
print(min_distance)

plt.imshow(np.reshape(faces_train[:,min_distance], (46,56)).T, cmap = 'gist_gray')
plt.xticks([]), plt.yticks([])
plt.show()