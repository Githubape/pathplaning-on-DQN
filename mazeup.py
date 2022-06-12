import random
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import cv2


def flager(flag, x1, x2, y1, y2, x):
    for i in range(x1, x2):
        for j in range(y1, y2):
            flag[i * x + j] = 0


def blocker(image, x , y, flag):
    print(image.shape)
    print(image.dtype)
 #   cv2.imshow("img",image)
 #   cv2.waitKey(500)
    num_rows = int(x/10)  # int(input("Rows: "))  # number of rows
    num_cols = int(y/10)  # int(input("Columns: "))  # number of columns

    # The array M is going to hold the array information for each cell.
    # The first four coordinates tell if walls exist on those sides
    # and the fifth indicates if the cell has been visited in the search.
    # M(LEFT, UP, RIGHT, DOWN, CHECK_IF_VISITED)
    M = np.zeros((int(num_rows), int(num_cols), 5), dtype=np.uint8)

    # The array image is going to be the output image to display
    #image = np.zeros((num_rows * 10, num_cols * 10, 3), dtype=np.float64)
    #cv2.imshow("img0",image)
    #cv2.waitKey(1000)
    # Set starting row and column
    r = 0
    c = 0
    history = [(r, c)]  # The history is the stack of visited locations

    # Trace a path though the cells of the maze and open walls along the path.
    # We do this with a while loop, repeating the loop until there is no history,
    # which would mean we backtracked to the initial start.
    while history:
        # random choose a candidata cell from the cell set histroy
        r, c = random.choice(history)
        M[r, c, 4] = 1  # designate this location as visited
        history.remove((r, c))
        check = []
        # If the randomly chosen cell has multiple edges
        # that connect it to the existing maze,
        if c > 0:
            if M[r, c - 1, 4] == 1:
                check.append('L')
            elif M[r, c - 1, 4] == 0:
                history.append((r, c - 1))
                M[r, c - 1, 4] = 2
        if r > 0:
            if M[r - 1, c, 4] == 1:
                check.append('U')
            elif M[r - 1, c, 4] == 0:
                history.append((r - 1, c))
                M[r - 1, c, 4] = 2
        if c < num_cols - 1:
            if M[r, c + 1, 4] == 1:
                check.append('R')
            elif M[r, c + 1, 4] == 0:
                history.append((r, c + 1))
                M[r, c + 1, 4] = 2
        if r < num_rows - 1:
            if M[r + 1, c, 4] == 1:
                check.append('D')
            elif M[r + 1, c, 4] == 0:
                history.append((r + 1, c))
                M[r + 1, c, 4] = 2

        # select one of these edges at random.
        if len(check):
            move_direction = random.choice(check)
            if move_direction == 'L':
                M[r, c, 0] = 1
                c = c - 1
                M[r, c, 2] = 1
            if move_direction == 'U':
                M[r, c, 1] = 1
                r = r - 1
                M[r, c, 3] = 1
            if move_direction == 'R':
                M[r, c, 2] = 1
                c = c + 1
                M[r, c, 0] = 1
            if move_direction == 'D':
                M[r, c, 3] = 1
                r = r + 1
                M[r, c, 1] = 1

    # Open the walls at the start and finish
    M[0, 0, 0] = 1
    M[num_rows - 1, num_cols - 1, 2] = 1

    # Generate the image for display
    for row in range(0, num_rows):
        for col in range(0, num_cols):
            cell_data = M[row, col]
            for i in range(10 * row + 2, 10 * row + 8):
                image[i, range(10 * col + 2, 10 * col + 8)] = 255
                flager(flag, i, i+1, 10*col+2, 10*col+8, x)
            if cell_data[0] == 1:
                image[range(10 * row + 2, 10 * row + 8), 10 * col] = 255
                flager(flag, 10*row+2, 10*row+8, 10*col, 10*col+1, x)
                image[range(10 * row + 2, 10 * row + 8), 10 * col + 1] = 255
                flager(flag, 10 * row + 2, 10 * row + 8, 10 * col + 1, 10*col+2, x)
            if cell_data[1] == 1:
                image[10 * row, range(10 * col + 2, 10 * col + 8)] = 255
                flager(flag, 10 * row, 10 * row + 1, 10 * col + 2, 10*col+8, x)
                image[10 * row + 1, range(10 * col + 2, 10 * col + 8)] = 255
                flager(flag, 10 * row + 1, 10 * row + 2, 10 * col + 2, 10*col+8, x)
            if cell_data[2] == 1:
                image[range(10 * row + 2, 10 * row + 8), 10 * col + 9] = 255
                flager(flag, 10 * row + 2, 10 * row + 8, 10 * col + 9, 10*col+10, x)
                image[range(10 * row + 2, 10 * row + 8), 10 * col + 8] = 255
                flager(flag, 10 * row + 2, 10 * row + 8, 10 * col + 8, 10*col+9, x)
            if cell_data[3] == 1:
                image[10 * row + 9, range(10 * col + 2, 10 * col + 8)] = 255
                flager(flag, 10 * row + 9, 10 * row + 10, 10 * col + 2, 10*col+8, x)
                image[10 * row + 8, range(10 * col + 2, 10 * col + 8)] = 255
                flager(flag, 10 * row + 8, 10 * row + 9, 10 * col + 2, 10*col+8, x)

    for i in range(x - 2, x):
        for j in range(y - 2, y):
            image[i][j][0] = 255
            image[i][j][1] = 255
            image[i][j][2] = 255
    flager(flag,x-2,x, y-2,y,x)

    for i in range(0, 2):
        for j in range(0, 2):
            image[i][j][0] = 255
            image[i][j][1] = 255
            image[i][j][2] = 255
    flager(flag,0,2,0,2,x)
    # Display the image
    #cv2.imshow("img",image)
    print(image.shape)
    print(image.dtype)
    # print(str(image.size))
    #cv2.waitKey(0)
    # plt.imshow(image, cmap=cm.Greys_r, interpolation='none')
    # plt.show()
