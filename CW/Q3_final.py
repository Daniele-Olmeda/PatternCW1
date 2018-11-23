#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%% PCA & LDA
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn

from collections import Counter 

#%% import faces and labels

faces_mat = sio.loadmat('face.mat') #load faces from file
faces = faces_mat['X']
labels = faces_mat['l']

#%% data partition
count = 0
faces_train = np.zeros((2576, 1))
faces_test = np.zeros((2576, 1))
labels_train = np.zeros((1, 1))
labels_test = np.zeros((1, 1))

for i in range(0,520):
    if count <8:
        faces_train = np.c_[faces_train, faces[:, i]] #attach first 8 vector image of each class to faces_train
        labels_train=np.c_[labels_train,labels[:,i]]
        count += 1
    elif count == 8:
        faces_test = np.c_[faces_test, faces[:, i]] #attach 9th vector image of each class to faces_test
        labels_test=np.c_[labels_test,labels[:,i]]
        count += 1
    elif count == 9:
        faces_test = np.c_[faces_test, faces[:, i]] #attach 10th vector image of each class to faces_test
        labels_test=np.c_[labels_test,labels[:,i]]
        count = 0
faces_test = faces_test[:,1:] #pop first zero vector
faces_train = faces_train[:,1:] #pop first zero vector
labels_train = labels_train[:,1:]
labels_test = labels_test[:,1:]

#%% 
average_face = faces_train.mean(1) #mean face
average_face_mat = np.tile(average_face, (416,1)).T #copy the mean face vector for each column in the matrix
A = faces_train - average_face_mat
covariance_mat = np.matmul(A, A.T)/416
eigvals, eigvecs = np.linalg.eigh(covariance_mat) #returns eigenvalues in order and normalized eigenvectors in the eigenvalues order

count = 0
for i in range (0,2576): #number of non-zero eigenvalues
    if eigvals[i] < 1:
        count += 1
#print("number of non-zero eigenvalues: %s" % (2576-count))

eigvals = np.flip(eigvals, 0) #flip eigenvalues vector
eigvecs = np.flip(eigvecs, 1) #flip eigenvectors matrix

#sum elementwise eigval, divide each element for the total sum (normalize) and see the influence of each eigevalue

#%% 
covariance_mat_ld = np.matmul(A.T, A)/416 #low dimensional covariance matrix
eigvals_ld, eigvecs_ld = np.linalg.eigh(covariance_mat_ld) #returns eigenvalues in order and normalized eigenvectors in the eigenvalues order

count = 0
for i in range(0,416): #number of non-zero eigenvalues
    if eigvals_ld[i] < 1:
        count += 1
#print("number of non-zero eigenvalues: %s" % (416-count))

eigvals_ld = np.flip(eigvals_ld, 0) #flip eigenvalues_ld vector
eigvecs_ld = np.flip(eigvecs_ld, 1) #flip eigenvectors_ld matrix

#%% Q2.a

#declare matrices
omega = np.zeros((416,1))
eigvecs_new = np.zeros((2576,416)) #these will be your low-dimensional eigenspace

#calculate u = Av and normalize u
for i in range(0,416):
    eigvecs_new[:,i] = np.matmul(A,eigvecs_ld[:,i])
    eigvecs_new[:,i] = eigvecs_new[:,i] / np.linalg.norm(eigvecs_new) #normalization

for i in range(0,416):
    omega[i,0] = np.matmul(A[:,0].T,eigvecs_new[:,i]) #The column of A you select corresponds to the image you want

omega_all = np.zeros((416,416))
for i in range(416): #calculate omega matrix with all the omegas of the training set
    for j in range(416):
        omega_all[j,i] = np.matmul(A[:,i].T,eigvecs_new[:,j]) #omega all will be a collection of individual omegas so the rows indicate encoding the vectors and the columns will be for each image

#%% At this point we have omega for each image, concatenated into a matrix

M_pca=200
M_lda=50

N_c=8;#8 data points per class in training set
N_train=416;#total number of data points in training set
N_test=104;#total number of data points in testing set
class_num=52;
count=1

#Overall mean
omega_all_reduced=omega_all[:M_pca,:] #only taking the number of rows of omega associated with M_pca
omega_avg =np.mean(omega_all[:M_pca,:],axis=1)

#Class means
class_means=np.zeros((M_pca,1))
class_data=np.zeros((M_pca,1))

for i in range (N_train):
    class_data=np.c_[class_data,omega_all_reduced[:,i]] #collecting one class of data in a vector
    if count%N_c==0: #concatenated 8 class_data columns
        class_data=class_data[:,1:]
        mean=(1/N_c)*(class_data.sum(axis=1))#sum function sums all elements in the above vector i.e. sums across rows
        class_means=np.c_[class_means,mean]
        class_data=np.zeros((M_pca,1))
    count+=1
class_means=class_means[:,1:] #contains 52 columns, each expressing a class mean, or average eigenface
#print(class_means)

#Finding Sw
scatter=np.zeros((M_pca,M_pca))
Sw=np.zeros((M_pca,M_pca))

j=0
index=0

for i in range (N_train):
    #intermediate step creating matrices used in scatter matrix formation
    v=(omega_all_reduced[:,i]-class_means[:,j])
    v.shape=(M_pca,1)
    v_T=(np.array([omega_all_reduced[:,i]-class_means[:,j]]))
    
    scatter_data=v.dot(v_T)
    scatter=scatter+scatter_data
    index+=1
    if index%N_c==0: 
        Sw=Sw+scatter #obtained Sw
        scatter=np.zeros((M_pca,M_pca))
        j+=1
        
#print(Sw.shape)

#Obtaining matrix Sb
Sb=np.zeros((M_pca,M_pca))
for i in range (class_num):
    v=(class_means[:,i]-omega_avg)
    v.shape=(M_pca,1)
    v_T=(np.array([class_means[:,i]-omega_avg]))
    
    Sb_data=v.dot(v_T)
    Sb=Sb+Sb_data

#print(Sb.shape)
np.linalg.matrix_rank(Sb) #C-1 (51)
#when reconstructing or using NN, use less than the rank of Sb

#Eigenvalues and eigenvectors
eigvals_s, eigvecs_s = np.linalg.eigh((np.linalg.inv(Sw)).dot(Sb)) #We yield M_lda number of highest eigenvalues
eigvals_s = np.flip(eigvals_s, 0) #flip eigenvalues_s vector - this has dimensions (416,) which we will reduce depending on significant eigenvalues
eigvecs_s = np.flip(eigvecs_s, 1) #flip eigenvectors_s matrix

#Calculating the classification space projetion matrix
rho=np.zeros((M_lda,N_train))
for i in range (N_train):
    rho[:,i]=np.matmul(((eigvecs_s[:,:M_lda]).T),omega_all_reduced[:,i])
    
#print(rho)
#print(rho.shape) #Has the shape M_lda*N_train

#Nearest Neighbour Classification 
def PCA_LDA_NN(test_img):
    omega_test = np.zeros((M_pca,1))
    test_face= faces_test[:, test_img] #take a test face
    test_face_minus_average=test_face - average_face
    for i in range(0,M_pca): #calculate omega of test image
        omega_test[i,0] = np.matmul(test_face_minus_average.T,eigvecs_new[:,i])

    #print(omega_test.shape) #(M_pca,1)
    
    #Projecting omega_test onto the classification space
    rho_test=np.zeros((M_lda,1))
    rho_test=np.matmul(((eigvecs_s[:,:M_lda]).T),omega_test)

    #print(rho_test.shape)

    #Classification by calculating minimum distance between rho_test and all rho values
    get_the_min=np.zeros((M_lda,N_train))
    for i in range(N_train):
        get_the_min[:,i]=rho[:,i]-rho_test[:,0]

    min_distance = 0
    for i in range(N_train):
        distance = np.linalg.norm(get_the_min[:,i])
        if distance <=  np.linalg.norm(get_the_min[:,min_distance]):
            min_distance = i
    
    accuracy=0
    x=labels_test[:,test_img]#true label
    y=labels_train[:,min_distance]#predicted label
    if x==y:
        accuracy=1
        
    return accuracy,x,y

#Evaluating classification accuracy using label data
result=0
label_true=[]#empty list, need list form to make confusion matrix 
label_predicted=[]
for i in range (N_test): #104 is number of test images
    accuracy,x,y=PCA_LDA_NN(i)
    result+=accuracy
    label_true.append(x)
    label_predicted.append(y)
    
#Confusion Matrix Calculation
array=confusion_matrix(label_true,label_predicted)
#print(array)
df_cm = pd.DataFrame(array, range(52),
                  range(52))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap='Greys')

#Percent accuracy
percent_accuracy=(result/104)*100
print(percent_accuracy)

print(label_true)
print(label_predicted)

#%% Randomizing training set (bagging) - THIS CODE WAS NOT USED, it involves random sampling with replacement of ALL data, rather than classes
#create array with 0,1,2,3....415 that we will use to refer to columns of faces_train
a=np.arange(416)
a_bagging=np.random.choice(a, 416,416)#samples with replacement

a_bagging.shape=(416,1)#forces shape from (416,) to (416,1)
#print(a_bagging)

faces_train_bag=np.zeros((2576,1))
labels_train_bag=np.zeros((1,1))
for i in range (416):
    x=a_bagging[i,0]
    faces_train_bag=np.c_[faces_train_bag,faces_train[:,x]]
    labels_train_bag=np.c_[labels_train_bag,labels_train[:,x]]
    
faces_train_bag=faces_train_bag[:,1:]
labels_train_bag=labels_train_bag[:,1:]

idx=labels_train_bag.argsort()[::1]#sorts "across"

#print(idx)

labels_train_bag=labels_train_bag[:,idx]#sorts labels
faces_train_bag=faces_train_bag[:,idx]#sorts data corresponding to labels

labels_train=labels_train_bag
faces_train=faces_train_bag

labels_train.shape=(1,416)
faces_train.shape=(2576,416)
#print(labels_train)
#print(faces_train)

#%% PCA & LDA redone with random sampling of feature space

#%% import faces and labels

faces_mat = sio.loadmat('face.mat') #load faces from file
faces = faces_mat['X']
labels = faces_mat['l']


#%% data partition
N_train=416;#total number of data points in training set
N_test=104;#total number of data points in testing set

count = 0
faces_train = np.zeros((2576, 1))
faces_test = np.zeros((2576, 1))
labels_train = np.zeros((1, 1))
labels_test = np.zeros((1, 1))

for i in range(0,520):
    if count <8:
        faces_train = np.c_[faces_train, faces[:, i]] #attach first 8 vector image of each class to faces_train
        labels_train=np.c_[labels_train,labels[:,i]]
        count += 1
    elif count == 8:
        faces_test = np.c_[faces_test, faces[:, i]] #attach 9th vector image of each class to faces_test
        labels_test=np.c_[labels_test,labels[:,i]]
        count += 1
    elif count == 9:
        faces_test = np.c_[faces_test, faces[:, i]] #attach 10th vector image of each class to faces_test
        labels_test=np.c_[labels_test,labels[:,i]]
        count = 0
faces_test = faces_test[:,1:] #pop first zero vector
faces_train = faces_train[:,1:] #pop first zero vector
labels_train = labels_train[:,1:]
labels_test = labels_test[:,1:]
#%% 
average_face = faces_train.mean(1) #mean face
average_face_mat = np.tile(average_face, (416,1)).T #copy the mean face vector for each column in the matrix
A = faces_train - average_face_mat
covariance_mat = np.matmul(A, A.T)/416
eigvals, eigvecs = np.linalg.eigh(covariance_mat) #returns eigenvalues in order and normalized eigenvectors in the eigenvalues order

count = 0
for i in range (0,2576): #number of non-zero eigenvalues
    if eigvals[i] < 1:
        count += 1
#print("number of non-zero eigenvalues: %s" % (2576-count))

eigvals = np.flip(eigvals, 0) #flip eigenvalues vector
eigvecs = np.flip(eigvecs, 1) #flip eigenvectors matrix

#%% 
covariance_mat_ld = np.matmul(A.T, A)/416 #low dimensional covariance matrix
eigvals_ld, eigvecs_ld = np.linalg.eigh(covariance_mat_ld) #returns eigenvalues in order and normalized eigenvectors in the eigenvalues order

count = 0
for i in range(0,416): #number of non-zero eigenvalues
    if eigvals_ld[i] < 1:
        count += 1
#print("number of non-zero eigenvalues: %s" % (416-count))

eigvals_ld = np.flip(eigvals_ld, 0) #flip eigenvalues_ld vector
eigvecs_ld = np.flip(eigvecs_ld, 1) #flip eigenvectors_ld matrix

#declare matrices
omega = np.zeros((416,1))
eigvecs_new = np.zeros((2576,416)) #these will be your low-dimensional eigenspace

#calculate u = Av and normalize u
for i in range(0,416):
    eigvecs_new[:,i] = np.matmul(A,eigvecs_ld[:,i])
    eigvecs_new[:,i] = eigvecs_new[:,i] / np.linalg.norm(eigvecs_new) #normalization

#Random sampling in feature space
M0=20 
M1=30  
M_pca=M0+M1#M0 is fixed as M0 largest eigenfaces, M1 are randomly selected from the remainining non-zero eigenfaces

eigvec_M0=eigvecs_new[:,:M0] 
#print(eigvec_M0.shape)
eigvecs_new_remain=eigvecs_new[:,M0:]
#print(eigvecs_new_remain.shape)

K=50 #number of subspaces whose outputs will be fused later
omega_all_K=np.zeros((M_pca,1))
for i in range (K):
    rand_sample=np.random.choice(416-M0,M1,replace=False)
    rand_sample.shape=(M1,1) #forces shape from (M1,) to (M1,1)

    eigvec_M1=np.zeros((2576,1))
    for i in range (M1):
        x=rand_sample[i,0]
        eigvec_M1=np.c_[eigvec_M1,eigvecs_new_remain[:,x]]
    eigvec_M1=eigvec_M1[:,1:]
    eigvecs_new=np.concatenate((eigvec_M0, eigvec_M1), axis=1)

    omega_all = np.zeros((M_pca,N_train))
    for i in range(N_train): #calculate omega matrix with all the omegas of the training set
        for j in range(M_pca):
            omega_all[j,i] = np.matmul(A[:,i].T,eigvecs_new[:,j]) #omega all will be a collection of individual omegas so the rows indicate encoding the vectors and the columns will be for each image
    
    omega_all_K=np.c_[omega_all_K,omega_all]

omega_all_K=omega_all_K[:,1:]

N_c=8;#8 data points per class in training set
class_num=52;

omega_all=np.zeros((M_pca,N_train))
omega_all_data=np.zeros((M_pca))

label_predicted_K=np.zeros((1,N_test))


index=1
for i in range (N_train*K):
    omega_all_data=np.c_[omega_all_data,omega_all_K[:,i]]
    if index%N_train==0:
        omega_all_data=omega_all_data[:,1:]
        omega_all=omega_all_data
        #Overall mean
        omega_all_reduced=omega_all[:M_pca,:] #i.e. only taking the number of rows of omega associated with M_pca
        omega_avg=np.mean(omega_all[:M_pca,:],axis=1)

        #Class means
        class_means=np.zeros((M_pca,1))
        class_data=np.zeros((M_pca,1))
        
        count=1
        for i in range (N_train):
            class_data=np.c_[class_data,omega_all_reduced[:,i]] #here we have collected one class of data in a vector
            if count%N_c==0: #meaning we have concatenated 8 class_data columns
                class_data=class_data[:,1:]
                mean=(1/N_c)*(class_data.sum(axis=1))#sum function sums all elements in the above vector i.e. sums across rows
                class_means=np.c_[class_means,mean]
                class_data=np.zeros((M_pca,1))
            count+=1
        class_means=class_means[:,1:]
        #print(class_means)
        
        #Finding Sw
        scatter=np.zeros((M_pca,M_pca))
        Sw=np.zeros((M_pca,M_pca))

        j=0
        increment=0

        for i in range (N_train):
            #intermediate step creating matrices used in scatter matrix formation
            v=(omega_all_reduced[:,i]-class_means[:,j])
            v.shape=(M_pca,1)
            v_T=(np.array([omega_all_reduced[:,i]-class_means[:,j]]))

            scatter_data=v.dot(v_T)
            scatter=scatter+scatter_data
            increment+=1
            if increment%N_c==0: 
                Sw=Sw+scatter #obtained Sw
                scatter=np.zeros((M_pca,M_pca))
                j+=1

        #print(Sw.shape)

        #Obtaining matrix Sb
        Sb=np.zeros((M_pca,M_pca))
        for i in range (class_num):
            v=(class_means[:,i]-omega_avg)
            v.shape=(M_pca,1)
            v_T=(np.array([class_means[:,i]-omega_avg]))

            Sb_data=v.dot(v_T)
            Sb=Sb+Sb_data

        #print(Sb.shape)
        np.linalg.matrix_rank(Sb) 

        #Eigenvalues and eigenvectors
        eigvals_s, eigvecs_s = np.linalg.eigh((np.linalg.inv(Sw)).dot(Sb)) #We yield M_lda number of highest eigenvalues
        eigvals_s = np.flip(eigvals_s, 0) #flip eigenvalues_s vector - this has dimensions (416,) which we will reduce depending on significant eigenvalues
        eigvecs_s = np.flip(eigvecs_s, 1) #flip eigenvectors_s matrix

        #Select Mlda
        M_lda=20
        
        #Calculating the classification space projetion matrix
        rho=np.zeros((M_lda,N_train))
        for i in range (N_train):
            rho[:,i]=np.matmul(((eigvecs_s[:,:M_lda]).T),omega_all_reduced[:,i])

        #print(rho)
        #print(rho.shape) #Has the shape M_lda*N_train

        #Nearest Neighbour Classification 
        #compare projected data ()
        def PCA_LDA_NN(test_img):
            omega_test = np.zeros((M_pca,1))
            test_face= faces_test[:, test_img] #take a test face
            test_face_minus_average=test_face - average_face
            for i in range(0,M_pca): #calculate omega of test image
                omega_test[i,0] = np.matmul(test_face_minus_average.T,eigvecs_new[:,i])

            #print(omega_test.shape) #(M_pca,1)

            #Projecting omega_test onto the classification space
            rho_test=np.zeros((M_lda,1))
            rho_test=np.matmul(((eigvecs_s[:,:M_lda]).T),omega_test)

            #print(rho_test.shape)

            #Classification by calculating minimum distance between rho_test and all rho values
            get_the_min=np.zeros((M_lda,N_train))
            for i in range(N_train):
                get_the_min[:,i]=rho[:,i]-rho_test[:,0]

            min_distance = 0
            for i in range(N_train):
                distance = np.linalg.norm(get_the_min[:,i])
                if distance <=  np.linalg.norm(get_the_min[:,min_distance]):
                    min_distance = i

            accuracy=0
            x=labels_test[:,test_img]#true label
            y=labels_train[:,min_distance]#predicted label
            if x==y:
                accuracy=1

            return accuracy,x,y

        #Evaluating classification accuracy using label data
        result=0
        label_true=np.zeros((1,N_test))#empty list, need list form to make confusion matrix 
        label_predicted=np.zeros((1,N_test))
        for i in range (N_test): #104 is number of test images
            accuracy,x,y=PCA_LDA_NN(i)
            result+=accuracy
            label_true[:,i]=int(x)
            label_predicted[:,i]=int(y)
        
        #Percent accuracy
        percent_accuracy=(result/104)*100
        #print(percent_accuracy)
        
        omega_all_data=np.zeros((M_pca))
        omega_all=np.zeros((M_pca,M_pca))
    
        label_predicted_K=np.concatenate((label_predicted_K,label_predicted),axis=0)
        #print(label_predicted)
    i+=1
    index+=1

label_predicted_K=label_predicted_K[1:,:]
label_predicted_K=label_predicted_K.astype(int) #convert elements from float to int
print(label_predicted_K)
#print(label_predicted_K.shape)#(10,104) i.e. 104 images and K=10 sets of testing

#Majority voting in ensemble model
majority=np.zeros((1,N_test))
from collections import Counter 

i=0 
for row in label_predicted_K.T:
    freq=Counter(row).most_common(1)[0][0]
    #print(freq)
    majority[:,i]=freq
    #print(x)
    i+=1
    
print(majority)

accuracy=0
for i in range (N_test):
    if majority[:,i]==label_true[:,i]:
        accuracy+=1

percent_accuracy=(accuracy/104)*100
print(percent_accuracy)

#%% Bagging algorithm which was used - Class related bagging

faces_mat = sio.loadmat('face.mat') #load faces from file
faces = faces_mat['X']
labels = faces_mat['l']


#%% data partition
count = 0
faces_train = np.zeros((2576, 1))
faces_test = np.zeros((2576, 1))
labels_train = np.zeros((1, 1))
labels_test = np.zeros((1, 1))

for i in range(0,520):
    if count <8:
        faces_train = np.c_[faces_train, faces[:, i]] #attach first 8 vector image of each class to faces_train
        labels_train=np.c_[labels_train,labels[:,i]]
        count += 1
    elif count == 8:
        faces_test = np.c_[faces_test, faces[:, i]] #attach 9th vector image of each class to faces_test
        labels_test=np.c_[labels_test,labels[:,i]]
        count += 1
    elif count == 9:
        faces_test = np.c_[faces_test, faces[:, i]] #attach 10th vector image of each class to faces_test
        labels_test=np.c_[labels_test,labels[:,i]]
        count = 0
faces_test = faces_test[:,1:] #pop first zero vector
faces_train = faces_train[:,1:] #pop first zero vector
labels_train = labels_train[:,1:]
labels_test = labels_test[:,1:]
#%% 
average_face = faces_train.mean(1) #mean face
average_face_mat = np.tile(average_face, (416,1)).T #copy the mean face vector for each column in the matrix
A = faces_train - average_face_mat
covariance_mat = np.matmul(A, A.T)/416
eigvals, eigvecs = np.linalg.eigh(covariance_mat) #returns eigenvalues in order and normalized eigenvectors in the eigenvalues order

count = 0
for i in range (0,2576): #number of non-zero eigenvalues
    if eigvals[i] < 1:
        count += 1

eigvals = np.flip(eigvals, 0) #flip eigenvalues vector
eigvecs = np.flip(eigvecs, 1) #flip eigenvectors matrix
#%% 
covariance_mat_ld = np.matmul(A.T, A)/416 #low dimensional covariance matrix
eigvals_ld, eigvecs_ld = np.linalg.eigh(covariance_mat_ld) #returns eigenvalues in order and normalized eigenvectors in the eigenvalues order

count = 0
for i in range(0,416): #number of non-zero eigenvalues
    if eigvals_ld[i] < 1:
        count += 1
print("number of non-zero eigenvalues: %s" % (416-count))

eigvals_ld = np.flip(eigvals_ld, 0) #flip eigenvalues_ld vector
eigvecs_ld = np.flip(eigvecs_ld, 1) #flip eigenvectors_ld matrix

#%% 

#declare matrices
omega = np.zeros((416,1))
eigvecs_new = np.zeros((2576,416)) #these will be your low-dimensional eigenspace

#calculate u = Av and normalize u
for i in range(0,416):
    eigvecs_new[:,i] = np.matmul(A,eigvecs_ld[:,i])
    eigvecs_new[:,i] = eigvecs_new[:,i] / np.linalg.norm(eigvecs_new) #normalization

for i in range(0,416):
    omega[i,0] = np.matmul(A[:,0].T,eigvecs_new[:,i]) #The column of A you select corresponds to the image you want

M_pca=200

omega_all = np.zeros((416,416))
for i in range(416): #calculate omega matrix with all the omegas of the training set
    for j in range(416):
        omega_all[j,i] = np.matmul(A[:,i].T,eigvecs_new[:,j]) #omega all will be a collection of individual omegas so the rows indicate encoding the vectors and the columns will be for each image

N_c=8;#8 data points per class in training set
N_train=416;#total number of data points in training set
N_test=104;#total number of data points in testing set
class_num=52;
count=1
L1=10 # Number of classes sampled

rand_sample=np.random.choice(class_num,L1,replace=False)
rand_sample.shape=(L1,1)
#print(rand_sample)

#Overall mean
omega_all_reduced=omega_all[:M_pca,:] #i.e. only taking the number of rows of omega associated with M_pca
omega_avg =np.mean(omega_all[:M_pca,:],axis=1)

#Class means
class_means=np.zeros((M_pca,1))
class_data=np.zeros((M_pca,1))

label_predicted_K=np.zeros((1,N_test))

K=10
for i in range (K):
    for i in range (L1):
        x=rand_sample[i][0]
        start_class=x*N_c
        print(start_class)
        for j in range(start_class,start_class+N_c):
            class_data=np.c_[class_data,omega_all_reduced[:,j]] #here we have collected one class of data in a vector
        class_data=class_data[:,1:]
        mean=(1/N_c)*(class_data.sum(axis=1))#sum function sums all elements in the above vector i.e. sums across rows
        class_means=np.c_[class_means,mean]
        class_data=np.zeros((M_pca,1))

    class_means=class_means[:,1:]
    #print(class_means)

    #Finding Sw
    scatter=np.zeros((M_pca,M_pca))
    Sw=np.zeros((M_pca,M_pca))

    s=0
    index=0

    for j in range(L1):
        for i in range (N_train):
            x=rand_sample[j][0]
            start_class=x*N_c
            if i==start_class:
                for z in range (start_class,start_class+N_c):
                    #intermediate step creating matrices used in scatter matrix formation
                    v=(omega_all_reduced[:,z]-class_means[:,j])
                    v.shape=(M_pca,1)
                    v_T=(np.array([omega_all_reduced[:,z]-class_means[:,j]]))

                    scatter_data=v.dot(v_T)
                    scatter=scatter+scatter_data
                    index+=1
                Sw=Sw+scatter #obtained Sw
                scatter=np.zeros((M_pca,M_pca))

    #print(Sw.shape)

    #Obtaining matrix Sb
    Sb=np.zeros((M_pca,M_pca))
    for i in range (L1):
        v=(class_means[:,i]-omega_avg)
        v.shape=(M_pca,1)
        v_T=(np.array([class_means[:,i]-omega_avg]))

        Sb_data=v.dot(v_T)
        Sb=Sb+Sb_data

    #print(Sb.shape)
    np.linalg.matrix_rank(Sb) #it is 51, i.e. C-1 
    #Note that the rank of Sb is the upper bound on M is C-1 - so when reconstructing or using NN, use less than that (i.e. we want to use less than C)

    #Eigenvalues and eigenvectors
    eigvals_s, eigvecs_s = np.linalg.eigh((np.linalg.inv(Sw)).dot(Sb)) #We yield M_lda number of highest eigenvalues
    eigvals_s = np.flip(eigvals_s, 0) #flip eigenvalues_s vector - this has dimensions (416,) which we will reduce depending on significant eigenvalues
    eigvecs_s = np.flip(eigvecs_s, 1) #flip eigenvectors_s matrix

    #Select Mlda
    M_lda=45
   
    #Calculating the classification space projetion matrix
    rho=np.zeros((M_lda,N_train))
    for i in range (N_train):
        rho[:,i]=np.matmul(((eigvecs_s[:,:M_lda]).T),omega_all_reduced[:,i])

    #Nearest Neighbour Classification 
    def PCA_LDA_NN(test_img):
        omega_test = np.zeros((M_pca,1))
        test_face= faces_test[:, test_img] #take a test face
        test_face_minus_average=test_face - average_face
        for i in range(0,M_pca): #calculate omega of test image
            omega_test[i,0] = np.matmul(test_face_minus_average.T,eigvecs_new[:,i])

        #print(omega_test.shape) #(M_pca,1)

        #Projecting omega_test onto the classification space
        rho_test=np.zeros((M_lda,1))
        rho_test=np.matmul(((eigvecs_s[:,:M_lda]).T),omega_test)

        #print(rho_test.shape)

        #Classification by calculating minimum distance between rho_test and all rho values
        get_the_min=np.zeros((M_lda,N_train))
        for i in range(N_train):
            get_the_min[:,i]=rho[:,i]-rho_test[:,0]

        min_distance = 0
        for i in range(N_train):
            distance = np.linalg.norm(get_the_min[:,i])
            if distance <=  np.linalg.norm(get_the_min[:,min_distance]):
                min_distance = i

        accuracy=0
        x=labels_test[:,test_img]#true label
        y=labels_train[:,min_distance]#predicted label
        if x==y:
            accuracy=1

        return accuracy,x,y

    #Evaluating classification accuracy using label data
    result=0
    label_true=np.zeros((1,N_test))#empty list, need list form to make confusion matrix 
    label_predicted=np.zeros((1,N_test))
    for i in range (N_test): #104 is number of test images
        accuracy,x,y=PCA_LDA_NN(i)
        result+=accuracy
        label_true[:,i]=int(x)
        label_predicted[:,i]=int(y)

    #Percent accuracy
    percent_accuracy=(result/104)*100
    print(percent_accuracy)
    #print(label_predicted_K.shape)
    
    #print(label_predicted.shape)
    label_predicted_K=np.concatenate((label_predicted_K,label_predicted),axis=0)
    

label_predicted_K=label_predicted_K[1:,:]
label_predicted_K=label_predicted_K.astype(int) #convert elements from float to int
#print(label_predicted_K)
#printing accuracy with varied M_LDA

majority=np.zeros((1,N_test))

i=0 
for row in label_predicted_K.T:
    freq=Counter(row).most_common(1)[0][0]
    #print(freq)
    majority[:,i]=freq
    #print(x)
    i+=1
    
#print(majority)

accuracy=0
for i in range (N_test):
    if majority[:,i]==label_true[:,i]:
        accuracy+=1

#percent_accuracy=(accuracy/104)*100
#print(percent_accuracy)


# In[ ]:




