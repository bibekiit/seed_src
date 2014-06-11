import numpy as np
import scipy.io as sio        #remove this later
import pdb


##def call():
##
##        #pdb.set_trace()
##
##    x = sio.loadmat('calcorient_problemcase.mat')
##    matrix_dict = {'A':x['A'],'B':x['B'],'rA':x['rA'],'rB':x['rB']}
##
##    output = calc_orient(matrix_dict['A'], matrix_dict['rA'], matrix_dict['B'], matrix_dict['rB'])
##
##    print output




##def call( ):
##
##    x = scipy.io.loadmat('A1.mat')
##    A = x['A']
##    x = scipy.io.loadmat('B1.mat')
##    B = x['B']
##    x = scipy.io.loadmat('rA1.mat')
##    rA = x['rA']
##    x = scipy.io.loadmat('rB1.mat')
##    rB = x['rB']
##
##    output = calc_orient(A,rA,B,rB)
##
##    print output
##    
    



def calc_orient(A, rA, B, rB):

    """A, B, rA, rB are numpy arrays of dimension 2. Output of this function
    is a list output, which has 3 elements. First and second entries are scalars
    called similarity and index respectively, third entry is a numpy array
    called r_g of length columnsize(A)."""
    
    

    #pdb.set_trace()
    raw_val = []

    noOfrows_A = A.shape[0]

    try:
        noOfcols_A = A.shape[1]
    except IndexError:
        noOfcols_A = 1


    index = 1
    r_g = []


    

    for i in range(noOfrows_A):

        if i >= rA.size or i >= rB.size:
            break
        
        count = 0
        #Following try-except block assigns value for any arbitrary index. If index is
        #beyond length of raw_val, zeros are added.
        try:
            raw_val[index - 1] = 0
        except IndexError:
            p = len(raw_val)
            while p < index - 1:
                raw_val.append(0)
                p = p + 1
            raw_val.append(0)

        for j in range(noOfcols_A):
            # The binary operator ~= in MATLAB returns 1 when its arguments dissimilar, else 0.
            if rA[i,j] != -1 and rB[i,j] != -1:
                
                temp = abs(A[i,j] - B[i,j])

                
                try:
                    raw_val[j] = temp       
                except IndexError:
                    p = len(raw_val)
                    while p < j:
                        raw_val.append(0)
                        p = p + 1
                    raw_val.append(temp)
                
                try:
                    r_g[count] = j + 1      #Because MATLAB follows a 1 based indexing
                except IndexError:
                    p = len(r_g)            #p is dummy loop variable
                    while p < count:
                        r_g.append(0)
                        p = p + 1
                    r_g.append(j + 1)       #Because MATLAB follows a 1 based indexing

                count = count + 1

        #print 'count = ' + str(count)
        

        if count > 0:
            index = count
        else:
            raw_val[index - 1] = 0           #it is not necessary to put that try-except block here
      

    raw_val = np.asarray(raw_val)               #converting list to numpy array for later use
    #print 'raw_val earlier = ',
    #print raw_val


    if index > 1:
        q = np.exp(max(0, 70 - count)*(-1/float(50)))
        #print 'count = ' + str(count)
        #print 'q = ' + str(q)
        raw_val = np.exp(-raw_val*8)*q
        #print 'raw_val = ',
        #print raw_val
        #print 'raw_val.shape = ' + str(raw_val.shape)
        similarity = np.mean(raw_val)
    else:
        similarity = -1

    r_g = np.asarray(r_g)

    output = []                                 #final o/p from the func, list output

    output.append(similarity)
    output.append(index)
    output.append(r_g)
    

    return output


    


        
    
