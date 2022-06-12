import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import cv2
import os
import shutil

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import time


class Timer:
    def __init__(self, func=time.perf_counter):
        self.elapsed = 0.0
        self._func = func
        self._start = None

    def start(self):
        if self._start is not None:
            raise RuntimeError('Already started')
        self._start = self._func()

    def stop(self):
        if self._start is None:
            raise RuntimeError('Not started')
        end = self._func()
        self.elapsed += end - self._start
        self._start = None

    def reset(self):
        self.elapsed = 0.0

    @property
    def running(self):
        return self._start is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def getBatchInput(inputs, start, batchSize):
    first = start
    start = start + batchSize
    end = start
    return inputs[first:end], start


def maxer(output):
    la = [output[0], output[1]]#,output[2],output[3]]
    ls = sorted(la)
    return la.index(ls[1]), la.index((ls[0]))


outputFolder = 'testOutput'
t= Timer()
t2= Timer()
if os.path.exists(outputFolder):
    shutil.rmtree(outputFolder)

os.mkdir(outputFolder)

imageSize = [30, 30, 3]

learningRate = 0.001
lr_decay_rate = 0.9
lr_decay_step = 2000

checkpointFile = 'NewCheckpoints/Checkpoint2.ckpt'
x = tf.placeholder(tf.float32, shape=[None, imageSize[0], imageSize[1], imageSize[2]])
W_conv1 = weight_variable([2, 2, 3, 10])
b_conv1 = bias_variable([10])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

W_conv2 = weight_variable([2, 2, 10, 20])
b_conv2 = bias_variable([20])

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

h_conv2_flat = tf.reshape(h_conv2, [-1, 30 * 30 * 20])
W_fc1 = weight_variable([30 * 30 * 20, 100])
b_fc1 = bias_variable([100])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([100, 4])
b_fc2 = bias_variable([4])

y_out = tf.matmul(h_fc1, W_fc2) + b_fc2

loss = tf.reduce_mean(tf.square(y_out), 1)
avg_loss = tf.reduce_mean(loss)

global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(learningRate,
                                global_step,
                                lr_decay_step,
                                lr_decay_rate,
                                staircase=True)

train_step = tf.train.AdamOptimizer(lr).minimize(avg_loss)

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, checkpointFile)

testImages = 200
limit = imageSize[0] * imageSize[1]
point = np.genfromtxt('testPoints.txt')
writer = tf.summary.FileWriter('loggertime')

num = 0
chang=[12]
for i in range(20,43):
    t.reset()
    t.start()
    index = 0
    outputImage = cv2.imread('testImages/image_' + str(i) + '_' + str(index) + '.png')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(0, 30, 1)
    Y = np.arange(0, 30, 1)
    X, Y = np.meshgrid(X, Y)
    blueimg = outputImage[:, :, 0]  # 需要哪个通道的三维图，选择哪个通道即可。

    for i in range(imageSize[0]):
        for j in range(imageSize[1]):
            blueimg[i][j] = outputImage[i][j][0]
            if (blueimg[i][j] == 255):
                blueimg[i][j] = 150
                blueimg[i][j] = blueimg[i][j] - (60 - i - j) * 0.5
            else:
                blueimg[i][j] = 50

    stter = 2
    dom = 100
    for i in range(imageSize[0]):
        for j in range(imageSize[1]):
            conn = 0
            if (blueimg[i][j] >= 120):
                for m in range(i - stter, i + stter + 1):
                    for n in range(j - stter, j + stter + 1):
                        if (m < imageSize[0] and m >= 0 and n >= 0 and n < imageSize[1]):
                            if (blueimg[m][n] == 50):
                                conn = conn + 1
            blueimg[i][j] = blueimg[i][j] - dom / 24 * conn

    cv2.imshow("blu", blueimg)
    cv2.waitKey(100)

    surf = ax.plot_surface(X, Y, blueimg, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(0, 255)  # z轴的取值范围
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    continue

    #cv2.imshow("env",outputImage)
    # cv2.waitKey(5000)
    lcc = 1

    while not index == limit - 1:

        t2.reset()
        t2.start()

      #  if num == 120:
        #    break
        print('index is %d' % index)
        inputImages = cv2.imread('testImages/image_' + str(i) + '_' + str(index) + '.png')

        inputs = []
        inputs.append(inputImages)
        inputs = np.array(inputs)
        output = sess.run(y_out, feed_dict={x: inputs})
        print(output[0])
        # if output[0][0] > output[0][1] :
        #	if (index+1)%imageSize[0] == 0 :
        #		print ('wrong grid entered') #border right flase
        #		break
        #
        #	index = index+1

        # else :
        #	index = index+25
        mo, moo = maxer(output[0])

        print(str(num) + "mo" + str(mo))
        if mo == 0:
            if (lcc != 2):
                index = index + 1
                moo = -1

                print("r")
        elif mo == 1:
            if (lcc != 3):
                index = index + imageSize[0]
                moo = -1
                print("d")
        elif mo == 2:
            if (lcc != 0):
                index = index - 1
                moo = -1
                print("l")
        else:
            if (lcc != 1):
                index = index - imageSize[0]
                moo = -1
                print("u")

        lcc = mo
        if (moo != -1):
            if moo == 0:
                index = index + 1
                print("2r")
            elif moo == 1:
                index = index + imageSize[0]
                print("2d")
            elif moo == 2:
                index = index - 1
                print("2l")
            else:
                index = index - imageSize[0]
                print("2u")
            lcc = moo

        if index >= limit or index < 0 or point[i * limit + index] == -100:  # hit
            print('wrong grid entered')
            t2.stop()
            break
        else:
            outputImage[int(index / imageSize[0])][index % imageSize[0]][0] = 255
            outputImage[int(index / imageSize[0])][index % imageSize[0]][1] = 0
            outputImage[int(index / imageSize[0])][index % imageSize[0]][2] = 0
            tesing = cv2.resize(outputImage, (500, 500))
            cv2.imshow("img", tesing)
            cv2.waitKey(200)
            t2.stop()
            summary2 = tf.Summary(value=[
            tf.Summary.Value(tag="time-step", simple_value=t2.elapsed)])
            writer.add_summary(summary2, num)
            num = num + 1




    t.stop()
    print("t:"+str(t.elapsed))
    summary = tf.Summary(value=[
        tf.Summary.Value(tag="time", simple_value=t.elapsed)])
    writer.add_summary(summary, i)

    if index == limit - 1:
        cv2.imwrite(os.path.join(outputFolder, 'image_' + str(i) + '.png'), outputImage)
writer.close()
