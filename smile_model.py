# 92x112 image size

from os import listdir
import numpy as np
from PIL import Image

dat = "Datasets/datasets.npy"
src = "Datasets/smile_face/"
des = "Datasets/"
ext = ".pgm"
label = ["smiling", "not smiling"]
model_name = "Model/smile.model"


#
#
# global dataset
# dataset = []
# f = open("smile.txt", "r")
# smile = []
# for line in f:
#     smile.append(int(line))
#
# for _x in listdir(src):
#     if _x != ".DS_Store":
#         tar = int(_x.split(".")[0])
#         if tar in smile:
#             dataset.append([[tar], [1, 0]])
#
# c = 0
# for _x in listdir(src):
#     if _x != ".DS_Store":
#         tar = int(_x.split(".")[0])
#         if tar not in smile:
#             if c < 132:
#                 dataset.append([[tar], [0, 1]])
#                 c += 1
#             else:
#                 break
#
# print(len(dataset))
#
# dataset = np.array(dataset)
# np.random.shuffle(dataset)
# np.save(dat, dataset)


def saveDatasets(dat):
    dataset = np.load(dat)
    np.random.shuffle(dataset)
    train = dataset[0:211]
    test = dataset[211:264]
    test_x = []
    test_y = []
    train_x = []
    train_y = []

    print("*********Reading Train Data*********")
    for y in train:
        img = Image.open(src + str(y[0][0]) + ext)
        img = np.asarray(img)
        img = img.reshape(img.shape + (1,))
        train_x.append(img)
        train_y.append(y[1])  # one hot vector
    print("*********Reading Test Data*********")
    for x in test:
        img = Image.open(src + str(x[0][0]) + ext)
        img = np.asarray(img)
        img = img.reshape(img.shape + (1,))
        test_x.append(img)
        test_y.append(x[1])
    print("*********Loaded*********")
    np.save(des + "train_x", train_x)
    np.save(des + "train_y", train_y)
    np.save(des + "test_x", test_x)
    np.save(des + "test_y", test_y)


def readDatasets(des):
    return np.load(des + "train_x.npy"), np.load(des + "train_y.npy"), np.load(des + "test_x.npy"), np.load(
        des + "test_y.npy")

def init():
    img_rows = 112
    img_col = 92
    img_channel = 1
    strides = 3

    import tflearn
    from tflearn.layers.core import input_data, fully_connected, dropout
    from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
    from tflearn.layers.estimator import regression
    global convnet
    convnet = input_data(shape=[None, img_rows, img_col, img_channel], name='input')
    convnet = conv_2d(convnet, 16, 3, strides=[1, strides, strides, 1], activation='relu')
    convnet = conv_2d(convnet, 32, 2, strides=[1, strides, strides, 1], activation='relu')
    convnet = conv_2d(convnet, 16, 2, strides=[1, strides, strides, 1], activation='relu')
    convnet = conv_2d(convnet, 32, 2, strides=[1, strides, strides, 1], activation='relu')
    convnet = conv_2d(convnet, 16, 2, strides=[1, strides, strides, 1], activation='relu')
    convnet = max_pool_2d(convnet, [1, strides, strides, 1])
    # convnet = dropout(convnet, 0.1)
    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, learning_rate=0.001, loss='categorical_crossentropy', optimizer='adam')
    global model
    model = tflearn.DNN(convnet, tensorboard_dir="./Graph")


    # for i in range(len(res)):
    #     print(np.argmax(res[i]), np.argmax(test_Y[i]))
    # print(test_Y)


def loadModel():
    model.load(model_name)


def train():
    batch_size = 5
    epochs = 50
    model.fit(train_X, train_Y, n_epoch=epochs, show_metric=True, batch_size=batch_size, run_id='smile',
              validation_set=[test_X, test_Y])
    save = input("Do you want to save the model? Y/N")
    if save == 'Y':
        model.save(model_name)
        print("Saved")


def predict(x):
    return np.argmax(model.predict(x))



print("Reading Dataset")
train_X, train_Y, test_X, test_Y = readDatasets(des)
print("Finished Reading Dataset")

init()
train()
