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
import time
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix


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

start = time.time() #start timer
covariance_mat = np.matmul(A, A.T)/416
eigvals, eigvecs = np.linalg.eigh(covariance_mat) #returns eigenvalues in order and normalized eigenvectors in the eigenvalues order
eigvals = np.flip(eigvals, 0) #flip eigenvalues vector
eigvecs = np.flip(eigvecs, 1) #flip eigenvectors matrix
end = time.time() #end timer
print(end - start)

#normalize eigenvalues
eigvals_norm = eigvals / np.linalg.norm(eigvals)
eigvals_divided_bu_sum = eigvals/np.sum(eigvals)
sum_first_values_eigvalues= np.sum(eigvals_divided_bu_sum[:4])
sum_last_values_eigvalues = np.sum(eigvals_divided_bu_sum[411:415])

plt.imshow(np.reshape(average_face, (46,56)).T, cmap = 'gist_gray') #print mean face
plt.xticks([]),plt.yticks([])
plt.show()

plt.imshow(np.reshape(eigvecs[:,499], (46,56)).T, cmap = 'gist_gray') #print mean face
plt.xticks([]),plt.yticks([])
plt.show()

count = 0
for i in range (2576): #number of non-zero eigenvalues
    if eigvals[i] < 1:
        count += 1
print("number of non-zero eigenvalues: %s" % (2576-count))
#print(np.linalg.matrix_rank(covariance_mat))


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
start = time.time() #start timer
covariance_mat_ld = np.matmul(A.T, A)/416 #low dimensional covariance matrix
eigvals_ld, eigvecs_ld = np.linalg.eigh(covariance_mat_ld) #returns eigenvalues in order and normalized eigenvectors in the eigenvalues order
eigvals_ld = np.flip(eigvals_ld, 0) #flip eigenvalues_ld vector
eigvecs_ld = np.flip(eigvecs_ld, 1) #flip eigenvectors_ld matrix
end = time.time() #end timer
print(end - start)

count = 0
for i in range(416): #number of non-zero eigenvalues
    if eigvals_ld[i] < 1:
        count += 1
print("number of non-zero eigenvalues: %s" % (416-count))



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
    omega[i,0] = np.matmul(A[:,213].T,eigvecs_new[:,i]) #change A column value to change image

#print original face
face = faces_train[:,213] #change faces_train column value to change image
plt.imshow(np.reshape(face, (46,56)).T, cmap = 'gist_gray')
plt.xticks([]), plt.yticks([])
plt.show()

sum = 0
for i in range(415): #face reconstruction
    to_add = omega[i,0]*eigvecs_new[:,i]
    sum += to_add
face_r = average_face + sum

#print reconstructed face
plt.imshow(np.reshape(face_r, (46,56)).T, cmap = 'gist_gray')
plt.xticks([]), plt.yticks([])
plt.show()

#%% Q2.b NN classifier
faces_test = faces_test.reshape(2576,104)
face_new = faces_test[:, 45] #get an image from the test set
face_minus_average = face_new - average_face #get the phi of the image

plt.imshow(np.reshape(face_new, (46,56)).T, cmap = 'gist_gray') #print image from the test set
plt.xticks([]), plt.yticks([])
plt.show()

omega_all = np.zeros((416,416))
for i in range(416): #calculate omega matrix with all the omegas of the training set
    for j in range(416):
        omega_all[j,i] = np.matmul(A[:,i].T,eigvecs_new[:,j]) 


omega_new = np.zeros((416,1))
for i in range(0,416): #calculate omega of test image
    omega_new[i,0] = np.matmul(face_minus_average.T,eigvecs_new[:,i])

get_the_min = omega_new - omega_all #find the image with the closest omega
distance = np.linalg.norm(get_the_min[:,:], axis=0)

        
plt.imshow(np.reshape(faces_train[:,np.argmin(distance)], (46,56)).T, cmap = 'gist_gray') #plot image with the closest omega
plt.xticks([]), plt.yticks([])
plt.show()


#%% accuracy of NN classifier

start = time.time() #start timer

number_of_basis = 416 #number of eigenvalues
accuracy = 0
true_NN = np.zeros((104))
predicted_NN = np.zeros((104)) 

omega_all = np.zeros((number_of_basis,416))
omega_all[:,:] = np.matmul(eigvecs_new[:,0:number_of_basis].T,A[:,:]) 
        
for k in range(104):
    
    face_new = faces_test[:, k] #get an image from the test set
    face_minus_average = face_new - average_face
      
    omega_new = np.zeros((number_of_basis,1))
    omega_new[:,0] = np.matmul(eigvecs_new[:,:].T, face_minus_average)  #calculate omega of test image
    
    get_the_min = omega_new - omega_all #find the image with the closest omega
    distance = np.linalg.norm(get_the_min[:,:], axis=0)
            
    if np.argmin(distance)//8 == k//2: #check if the image found is from the right class
        accuracy+= 1
        true_NN[k] = k//2
    
    predicted_NN[k] = k//2

end = time.time() #end timer

print(accuracy/104*100)
print(end - start)

confusion_NN = confusion_matrix(true_NN, predicted_NN) #calculate confusion matrix

df_cm = pd.DataFrame(confusion_NN, range(52),range(52))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap='Greys')
#%% Q2.b alternative method

start = time.time() #start timer

#declare matrices
all_eigvals = np.zeros((52,8,1))
all_eigvecs = np.zeros((52,8,8))
B_all = np.zeros((52,2576,8))
average_all = np.zeros((52,2576,1))
difference = np.zeros((1,52))

for i in range(52):
    
    average_face_AM = faces_train[:,i*8:i*8+8].mean(1) #mean face
    average_face_AM = average_face_AM.reshape(2576,1)
    average_all[i,:,:] = average_face_AM
    
    average_face_mat_AM = np.tile(average_face_AM, (1,8)) #copy the mean face vector for each column in the matrix
    B = faces_train[:,i*8:i*8+8] - average_face_mat_AM
    B_all[i,:,:] = B #stack the B matrices toghter in a 3D matrix
    covariance_mat_ld_AM = np.matmul(B.T, B)/8 #low dimensional covariance matrix
    
    eigvals_ld_AM, eigvecs_ld_AM = np.linalg.eigh(covariance_mat_ld_AM) #returns eigenvalues in order and normalized eigenvectors in the eigenvalues order
    eigvals_ld_AM = np.flip(eigvals_ld_AM, 0) #flip eigenvalues_ld_AM vector
    eigvecs_ld_AM = np.flip(eigvecs_ld_AM, 1) #flip eigenvectors_ld_AM matrix
    
    eigvals_ld_AM = eigvals_ld_AM.reshape(8,1)
    all_eigvals[i,:,:] = eigvals_ld_AM #stack eigenvalues toghter in a 3D matrix
    all_eigvecs[i,:,:] = eigvecs_ld_AM #stack eigenvectors toghter in a 3D matrix
    

#declare matrices
omega_AM = np.zeros((416,52))
eigvecs_new_AM = np.zeros((2576,8))
faces_test_phi = np.zeros((52,2576,1))
reconstructed = np.zeros((52,2576,1))
faces_test = faces_test.reshape(1,2576,104)
difference = np.zeros((52))

true = np.zeros((104))
predicted = np.zeros((104)) 

accuracy_AM = 0
for face_number in range(104):
    
    faces_test_phi[:,:,0] = faces_test[0,:,face_number] - average_all[:,:,0] #get phi
    
    for i in range(52):
    
       #calculate u = Av and normalize u
        for m in range(8):
            eigvecs_new_AM[:,m] = np.matmul(B_all[i,:,:],all_eigvecs[i,:,m])
            eigvecs_new_AM[:,m] = eigvecs_new_AM[:,m] / np.linalg.norm(eigvecs_new_AM) #normalization
        
        for j in range(8): #calculate omega
            omega_AM[j,i] = np.matmul(faces_test_phi[i,:,0].T,eigvecs_new_AM[:,j]) #change A column value to change image
      
        sum = 0
        for k in range(8): #face reconstruction
            to_add = omega_AM[k,i]*eigvecs_new_AM[:,k]
            sum += to_add
        reconstructed[i,:,0] = average_all[i,:,0] + sum
            
        difference [i] = np.linalg.norm(faces_test[0,:,face_number] - reconstructed[i,:,0])

    if np.argmin(difference) == face_number//2: #find the image with the closest reconstruction
        accuracy_AM += 1
        true[face_number] = face_number//2 
        
    predicted[face_number] = face_number//2

    
    #print original face
#    face = faces_test[0,:,face_number] #change A column value to change image #switch row with column with different partition alghoritm
#    plt.imshow(np.reshape(face, (46,56)).T, cmap = 'gist_gray')
#    plt.xticks([]), plt.yticks([])
#    plt.show()
#    
#    
#    #print reconstructed face
#    min_distance = np.argmin(difference)
#    plt.imshow(np.reshape(reconstructed[min_distance,:,0], (46,56)).T, cmap = 'gist_gray')
#    plt.xticks([]), plt.yticks([])
#    plt.show()
#                
end = time.time() #end timer

print(accuracy_AM/104*100)
print(end - start)

confusion_AM = confusion_matrix(true, predicted) #calculate confusion matrix

df_cm = pd.DataFrame(confusion_AM, range(52),range(52))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap='Greys')
    

            
