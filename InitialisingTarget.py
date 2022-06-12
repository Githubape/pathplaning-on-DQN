#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import cv2
import os
import shutil

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.05) #截断的产生正态分布的随机数，即随机数与均值的差值若大于两倍的标准差，则重新生成
	return tf.Variable(initial)  #创建变量

def bias_variable(shape):
	initial = tf.constant(0.05, shape=shape) #创建常量
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def getBatchInput(inputs,start,batchSize) :
	first = start
	start = start + batchSize
	end = start
	return inputs[first:end],start 	

imageSize = [30,30,3]
batchSize = 100
games = 150
totalImages = games*imageSize[0]*imageSize[1] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
learningRate = 0.001
lr_decay_rate = 0.9
lr_decay_step = 2000

folderName = 'NewCheckpoints'

if os.path.exists(folderName):
	shutil.rmtree(folderName)

os.mkdir(folderName)

checkpointFile = 'NewCheckpoints/Checkpoint3.ckpt'
#输入 25 25 3
x = tf.placeholder(tf.float32, shape=[None, imageSize[0], imageSize[1], imageSize[2]])#在代码层面，每一个tensor值在graph上都是一个op，当我们将train数据分成一个个minibatch然后传入网络进行训练时，每一个minibatch都将是一个op，这样的话，一副graph上的op未免太多，也会产生巨大的开销；于是就有了tf.placeholder()，我们每次可以将 一个minibatch传入到x = tf.placeholder(tf.float32,[None,32])上，下一次传入的x都替换掉上一次传入的x，这样就对于所有传入的minibatch x就只会产生一个op，不会产生其他多余的op，进而减少了graph的开销。
#conv1
W_conv1 = weight_variable([2, 2, 3, 10]) #卷积核 2*2 ×3 10个
b_conv1 = bias_variable([10])    #bias 10

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1) #将输入小于0的值幅值为0，输入大于0的值不变。   padding same
#输出 25 25 10

#conv2
W_conv2 = weight_variable([2, 2, 10, 20]) #卷积核2*2 *10 20个
b_conv2 = bias_variable([20])

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
#输出25 25 20

#flat1
h_conv2_flat = tf.reshape(h_conv2, [-1, 30*30*20])#函数用于对输入tensor进行维度调整，但是这种调整方式并不会修改内部元素的数量以及元素之间的顺序，换句话说，reshape函数不能实现类似于矩阵转置的操作
W_fc1 = weight_variable([30*30*20, 100])#tensor的第一维度表示第二层卷积层的输出，大小为25*25带有20个filters，第二个参数是层中的神经元数量，我们可自由设置。
b_fc1 = bias_variable([100])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
#全连接层fc1，输入有252520=12500个神经元结点，输出有100个结点，则一共需要252520*100=1250000个权值参数W和100个偏置参数b

W_fc2 = weight_variable([100, 4])
b_fc2 = bias_variable([4])

y_out=tf.matmul(h_fc1, W_fc2) + b_fc2

loss = tf.reduce_mean(tf.square(y_out),1)
avg_loss = tf.reduce_mean(loss)

global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(learningRate,   #固定的学习率总是显得笨拙：太小速度太慢，太大又担心得不到最优解。一个很直接的想法就是随着训练的进行，动态设置学习率——随着训练次数增加，学习率逐步减小。而tf.train.exponential_decay()就是tf内置的一个生成动态减小学习率的函数。
								global_step,
								lr_decay_step,
								lr_decay_rate,
								staircase=True)

train_step = tf.train.AdamOptimizer(lr).minimize(avg_loss)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())

inputs = np.zeros([totalImages,imageSize[0],imageSize[1],imageSize[2]])
print ('reading inputs')
for i in range(totalImages) :
	temp = imageSize[0]*imageSize[1]
	print(i)
	inputs[i] = cv2.imread('trainImages/image_'+str(int(i/temp))+'_'+str(i%temp)+'.png')


print ('inputs read')

start = 0
initialTarget = []
iterations = int(totalImages/batchSize)
save_path = saver.save(sess, checkpointFile)
print("Model saved in file: %s" % save_path)

print('number of iterations is %d'%iterations) #1250
for i in range(iterations) :
	batchInput,start = getBatchInput(inputs,start,batchSize)
	
	batchOutput = sess.run(y_out,feed_dict={x: batchInput})
	if i%50 == 0 :
		print('%d iterations reached'%i)
	for j in range(batchSize) :
		initialTarget.append(batchOutput[j])

print (start)
np.savetxt('Targets200_New.txt',initialTarget)		

