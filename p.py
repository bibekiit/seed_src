def p(img,x,y,i):

    """img is a 2D numpy array. Output is a list [j,X,Y], where j is a scalar"""

    if i == 1 or i == 9:

        Y = y
        X = x + 1
        j = img[y - 1,x]

    elif i == 2:
        Y = y - 1
        X = x + 1
        j = img[y - 2,x]

    elif i == 3:
        Y = y - 1
        X = x
        j = img[y - 2,x - 1]

    elif i == 4:
        Y = y - 1
        X = x - 1
        j = img[y - 2,x - 2]

    elif i == 5:
        Y = y
        X = x - 1
        j = img[y - 1,x - 2]

    elif i == 6:
        Y = y + 1
        X = x - 1 
        j = img[y,x - 2]
         
    elif i == 7:
        Y = y + 1
        X = x
        j = img[y,x - 1]
         
    elif i == 8:
        Y = y + 1
        X = x + 1 
        j = img[y,x]

    return j,X,Y
