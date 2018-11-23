#%% show image of faces
for i in range(0, 520):
    face = faces[:,i]
    plt.subplot(52,10,i+1)
    plt.imshow(np.reshape(face, (46,56)).T, cmap = 'gist_gray')
    #plt.title('Face %d' % (i))
    plt.xticks([]), plt.yticks([])
    
#%% data partition
faces_test, faces_train, labels_test, labels_train = train_test_split(faces, labels, test_size=0.8, stratify = labels)
faces_test = faces_test.T
faces_train = faces_train.T
labels_test = labels_test.T
labels_train = labels_train.T