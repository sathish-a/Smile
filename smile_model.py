# 92x112 image size

from os import listdir
import numpy as np
from PIL import Image

dat = "Datasets/datasets.npy"
src = "Datasets/smile_face/"
ext = ".pgm"
label = ["smiling", "not smiling"]
model_name = "model"


#
# dataset = []
# f = open(src, "r")
# smile = []
# for line in f:
#     smile.append(int(line))
#
# print(smile)
#
# for _x in listdir(des):
#     if _x != ".DS_Store":
#         tar = int(_x.split(".")[0])
#         if tar in smile:
#             dataset.append([[tar], [1, 0]])
#         else:
#             dataset.append([[tar], [0, 1]])

# dataset = np.array(dataset)


def readDatasets(dat):
    dataset = np.load(dat)
    np.random.shuffle(dataset)
    train = dataset[0:350]
    test = dataset[350:400]
    test_x = []
    test_y = []
    train_x = []
    train_y = []

    print("*********Reading Train Data*********")
    for y in train:
        img = Image.open(src + str(y[0][0]) + ext).convert('LA')
        train_x.append(np.asarray(img))
        train_y.append(y[1])
    print("*********Reading Test Data*********")
    for x in test:
        img = Image.open(src + str(x[0][0]) + ext).convert('LA')
        test_x.append(np.asarray(img))
        test_y.append(x[1])
    print("*********Loaded*********")
    return train_x, train_y, test_x, test_y


train_X, train_Y, test_X, test_Y = readDatasets(dat)

print(train_X[0].shape)



#hyper paramaetes
batch_size = 50
epochs = 25
img_rows = 112
img_col = 92
img_channel = 2



import tflearn
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.estimator import regression


convnet = input_data(shape=[None, img_rows, img_col, img_channel], name='input')

convnet = conv_2d(convnet, 512, 2, strides=[1, 2, 2, 1], activation='relu')
convnet = max_pool_2d(convnet, [1, 2, 2, 1])

convnet = conv_2d(convnet, 256, 2, strides=[1, 2, 2, 1], activation='relu')
convnet = max_pool_2d(convnet, [1, 2, 2, 1])

convnet = conv_2d(convnet, 128, 2, strides=[1, 2, 2, 1], activation='relu')
convnet = avg_pool_2d(convnet, [1, 2, 2, 1])

convnet = conv_2d(convnet, 64, 2, strides=[1, 2, 2, 1], activation='relu')
convnet = max_pool_2d(convnet, [1, 2, 2, 1])
convnet = dropout(convnet, 0.5)

convnet = conv_2d(convnet, 32, 2, strides=[1, 2, 2, 1], activation='relu')
convnet = max_pool_2d(convnet, [1, 2, 2, 1])

convnet = conv_2d(convnet, 16, 2, strides=[1, 2, 2, 1], activation='relu')
convnet = max_pool_2d(convnet, [1, 2, 2, 1])

convnet = conv_2d(convnet, 8, 2, strides=[1, 2, 2, 1], activation='relu')
convnet = avg_pool_2d(convnet, [1, 2, 2, 1])
convnet = dropout(convnet, 0.8)

# convnet = conv_2d(convnet, 128, 2, strides=[1, 2, 2, 1], activation='relu')
# convnet = max_pool_2d(convnet, [1, 2, 2, 1])
# convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, learning_rate=0.001, loss='categorical_crossentropy', optimizer='adam')

model = tflearn.DNN(convnet)
model.fit(train_X, train_Y, n_epoch=epochs, show_metric=True, snapshot_step=20, batch_size=batch_size, run_id='cat_dog')