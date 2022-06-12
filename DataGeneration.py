import cv2
import numpy as np
import random
import CheckingConnectivity
import os
import shutil
import  mazeup
numberOfBlocks = 75
gridSize = [30, 30, 3]
totalGames = 150
#points = np.zeros([totalGames,gridSize[0]*gridSize[1]])
points = []
def assignBlocks(grid,xlimit,limit,numberOfBlocks,flag) :
	
	x1 = random.sample(range(1,limit-1),numberOfBlocks)  #随机障碍点 不包括起点终点
	for i in range(numberOfBlocks) :
		grid[int(x1[i]/xlimit)][x1[i]%xlimit][0] = 0
		grid[int(x1[i]/xlimit)][x1[i]%xlimit][1] = 0
		grid[int(x1[i]/xlimit)][x1[i]%xlimit][2] = 0	
	while not CheckingConnectivity.Checking(grid[:,:,0]) : #合格性test  rebuild if false ??????????????????????
		#print 'not Valid Connection'
		for i in range(numberOfBlocks) :
			grid[int(x1[i]/xlimit)][x1[i]%xlimit][0] = 255
			grid[int(x1[i]/xlimit)][x1[i]%xlimit][1] = 255
			grid[int(x1[i]/xlimit)][x1[i]%xlimit][2] = 255
		x1 = random.sample(range(1,limit-1),numberOfBlocks)
		for i in range(numberOfBlocks) :
			grid[int(x1[i]/xlimit)][x1[i]%xlimit][0] = 0
			grid[int(x1[i]/xlimit)][x1[i]%xlimit][1] = 0
			grid[int(x1[i]/xlimit)][x1[i]%xlimit][2] = 0	

	for i in x1 :
		flag[i] = 1


folderName = 'trainImages'

if os.path.exists(folderName):
	shutil.rmtree(folderName)

os.mkdir(folderName)
num = 0
for t in range(totalGames) :
	print("::::::::::::::::::::::"+str(t))
	if t%10 == 0 :
		print('%d steps reached'%t)
	a = np.zeros(gridSize)
	x,y,z = a.shape

	'''
	for i in range(x) :
		for j in range(y) :
			for k in range(z) :
				a[i][j][k] = 255
	'''

	#cv2.imshow('image',a)
	

	limit = x*y
	flag = np.zeros([limit])  #一维障碍点
	flag.fill(1)
	####assignBlocks(a,x,limit,numberOfBlocks,flag)  #build block
	mazeup.blocker(a, x, y, flag)
	#cv2.imshow('imageo', a)
	#cv2.waitKey(0)


	#cv2.imshow("flag",np.array(flag).reshape(50,50))
	#cv2.waitKey(0)

	for i in range(limit) :   #构造point 用于存txt 便于访问 reward
		if not flag[i] == 0 :
			#points[t][i] = -100
			points.append(-100) #hit block -100
		elif i== limit-1 :
			points.append(100)   #reach final +100
		else :
			points.append(0)		#no reward

	#points[t][limit-1] = 100
	# #final start build
	for i in range(x-2, x):
		for j in range(y-2,y):
			a[i][j][0]= 0
			a[i][j][1]= 255
			a[i][j][2]= 0
	#a[x-1][y-1][0] = 40
	#a[x - 2][y - 1][0] = 40
	#a[range(x-2, x)][range(y-2, y)][1] = 222
	#a[range(x-2, x)][range(y-2, y)][2] = 20

	#cv2.imshow('imageo2', a)
	#cv2.waitKey(0)

	for i in range(x) :   # statue point build  25*25statue*200game
		for j in range(y) :
			#if flag[i*x+j] == 0:
				a[i][j][0] = 255
				a[i][j][1] = 0
				a[i][j][2] = 0
				#cv2.imshow('imagei', a)
				#cv2.waitKey(200)
				print ("save"+str(t)+str(i)+str(j))
				cv2.imwrite(os.path.join(folderName, "image_"+str(t)+"_"+str(gridSize[0]*i+j)+".png"), a)
				#print(str(t)+"--"+str(num))
				#num = num+1
				if flag[i*x+j] == 0 :   #back step
					a[i][j][0] = 255
					a[i][j][1] = 255
					a[i][j][2] = 255
				else :
					a[i][j][0] = 0
					a[i][j][1] = 0
					a[i][j][2] = 0
	
	
np.savetxt('pointsNew.txt',points)

	

