from __future__ import division
import numpy as np
from scipy.io import loadmat,savemat
import os,pdb

def hminired(A):
    ##HMINIRED Initial reduction of cost matrix for the Hungarian method.
    ##B=assredin(A)
    ##A - the unreduced cost matris.
    ##B - the reduced cost matrix with linked zeros in each row.

    # Adapted from the Matlab code of Nicholas Borlin
    # 
    (m,n) = A.shape

    #Subtract column-minimum values from each column.
    colMin = A.min(axis = 0)
    A = A - np.dot(np.ones((n,1)), np.tile(colMin,(1,1)))

    # Subtract row-minimum values from each row.
    rowMin = A.min(axis = 1)
    A = A - np.dot(np.tile(rowMin,(1,1)).T, np.ones((1,n)))

    # Get positions of all zeros.
    (j,i) = np.nonzero(A.T==0)
    i = np.tile(i,(1,1)).T
    j = np.tile(j,(1,1)).T

    # Extend A to give room for row zero list header column.
    B = np.zeros((m, n+1))
    B[0:m,0:n] = A.copy()
    for k in range(0,n):
        # Get all column in this row. 
        cols=j[k==i].T
        # Insert pointers in matrix.
        B[k,np.hstack([np.array([n]),cols])] = np.hstack([-(cols+1),np.array([0])])
    return B
                                                         
def hminiass(A):
    #HMINIASS Initial assignment of the Hungarian method.
    #
    #[B,C,U]=hminiass(A)
    #A - the reduced cost matrix.
    #B - the reduced cost matrix, with assigned zeros removed from lists.
    #C - a vector. C(J)=I means row I is assigned to column J,
    #              i.e. there is an assigned zero in position I,J.
    #U - a vector with a linked list of unassigned rows.
    (n,np1) = A.shape

    # Initalize return vectors.
    C = np.zeros((1,n))
    U = np.zeros((1,n+1))

    # Initialize last/next zero "pointers".
    LZ = np.zeros((1,n))
    NZ = np.zeros((1,n))

    for i in range(1,n+1):
        # Set j to first unassigned zero in row i.
        lj=n+1
        j=-A[i-1,lj-1]

        # Repeat until we have no more zeros (j==0) or we find a zero
	# in an unassigned column (c(j)==0).
    
        while (C[0,j-1]!=0):
            # Advance lj and j in zero list.
            lj=j
            j=-A[i-1,lj-1]
    
            # Stop if we hit end of list.
            if (j==0):
                break
		

        if (j!=0):
            # We found a zero in an unassigned column.
            
            # Assign row i to column j.
            C[0,j-1]=i
            
            # Remove A(i,j) from unassigned zero list.
            A[i-1,lj-1]=A[i-1,j-1]

            # Update next/last unassigned zero pointers.
            NZ[0,i-1]=-A[i-1,j-1]
            LZ[0,i-1]=lj

            # Indicate A(i,j) is an assigned zero.
            A[i-1,j-1]=0
        else:
            # We found no zero in an unassigned column.

            # Check all zeros in this row.

            lj = n+1
            j = -A[i-1,lj-1]
            
            # Check all zeros in this row for a suitable zero in another row.
            while (j!=0):
                # Check the in the row assigned to this column.
                r=C[0,j-1]
                
                # Pick up last/next pointers.
                lm=LZ[0,r-1]
                m=NZ[0,r-1]
                
                # Check all unchecked zeros in free list of this row.
                while (m!=0):
                    # Stop if we find an unassigned column.
                    if (C[0,m-1]==0):
                        break
                                        
                    # Advance one step in list.
                    lm=m
                    m=-A[r-1,lm-1]
                                
                if (m==0):
                    # We failed on row r. Continue with next zero on row i.
                    lj=j
                    j=-A[i-1,lj-1]
                else:
                    # We found a zero in an unassigned column.
                    # Replace zero at (r,m) in unassigned list with zero at (r,j)
                    A[r-1,lm-1]=-j
                    A[r-1,j-1]=A[r-1,m-1]
            
                    # Update last/next pointers in row r.
                    NZ[0,r-1]=-A[r-1,m-1]
                    LZ[0,r-1]=j
            
                    # Mark A(r,m) as an assigned zero in the matrix . . .
                    A[r-1,m-1]=0
            
                    # ...and in the assignment vector.
                    C[0,m-1]=r
            
                    # Remove A(i,j) from unassigned list.
                    A[i-1,lj-1]=A[i-1,j-1]
            
                    # Update last/next pointers in row r.
                    NZ[0,i-1]=-A[i-1,j-1]
                    LZ[0,i-1]=lj
            
                    # Mark A(r,m) as an assigned zero in the matrix . . .
                    A[i-1,j-1]=0
            
                    # ...and in the assignment vector.
                    C[0,j-1]=i
                    
                    # Stop search.
                    break;
    # Create vector with list of unassigned rows.

    # Mark all rows have assignment.
    r=np.zeros((1,n))
    rows=C[C!=0].astype('int')
    r[0,rows-1]=rows
    empty=np.nonzero(r==0)[1] + 1

    # Create vector with linked list of unassigned rows.
    U=np.zeros((1,n+1))
    U[0,np.hstack([np.array([n+1])-1, empty-1])]=np.hstack([empty, np.array([0])])
    return [A, C, U]

def hmreduce(A,CH,RH,LC,LR,SLC,SLR):
    #HMREDUCE Reduce parts of cost matrix in the Hungerian method.
    #
    #[A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR)
    #Input:
    #A   - Cost matrix.
    #CH  - vector of column of 'next zeros' in each row.
    #RH  - vector with list of unexplored rows.
    #LC  - column labels.
    #RC  - row labels.
    #SLC - set of column labels.
    #SLR - set of row labels.
    #
    #Output:
    #A   - Reduced cost matrix.
    #CH  - Updated vector of 'next zeros' in each row.
    #RH  - Updated vector of unexplored rows.

    n = A.shape[0]

    # Find which rows are covered, i.e. unlabelled.
    coveredRows=LR==0

    # Find which columns are covered, i.e. labelled.
    coveredCols=LC!=0

    r = np.nonzero(~coveredRows)[1] + 1
    c = np.nonzero(~coveredCols)[1] + 1

    # Get minimum of uncovered elements.
    m = np.amin(A[np.ix_(r-1,c-1)])

    # Subtract minimum from all uncovered elements.
    A[np.ix_(r-1,c-1)]=A[np.ix_(r-1,c-1)]-m
                
    
                                    
    # Check all uncovered columns..
    for j in c:
        # ...and uncovered rows in path order..
        for i in SLR:
            # If this is a (new) zero..
            if A[i-1,j-1]==0:
                # If the row is not in unexplored list..
                if (RH[0,i-1]==0):
                    # ...insert it first in unexplored list.
                    RH[0,i-1] = RH[0,n]
                    RH[0,n] = i
                    # Mark this zero as "next free" in this row.
                    CH[0,i-1] = j
                # Find last unassigned zero on row I.
                row = A[i-1,:]
                colsInList=-row[row<0]
                if (len(colsInList)==0):
                    # No zeros in the list.
                    l =n+1
                else:
                    for x in colsInList:
                        if (row[x-1]==0):
                            l=x
                # Append this zero to end of list.
                A[i-1,l-1] = -j
    
    # Add minimum to all doubly covered elements.
    r = np.nonzero(coveredRows)[1] + 1
    c = np.nonzero(coveredCols)[1] + 1   
    
    # Take care of the zeros we will remove.
    (i,j) = np.nonzero(A[np.ix_(r-1,c-1)]<=0)
    i = i+1
    j = j+1

    i = r[i-1]
    j = c[j-1]

    for k in range(1,len(i)+1):
        # Find zero before this in this row.
        lj = np.nonzero(A[i[k-1]-1,:]==-j[k-1])[0]+1
        # Link past it.
        A[np.ix_(np.array([i[k-1]])-1,lj-1)] = A[i[k-1]-1,j[k-1]-1]
        #  Mark it as assigned.
        A[i[k-1]-1,j[k-1]-1] = 0

    A[np.ix_(r-1,c-1)]=A[np.ix_(r-1,c-1)]+m

    return [A,CH,RH]

def hmflip(A,C,LC,LR,U,l,r):
    #HMFLIP Flip assignment state of all zeros along a path.
    #
    #[A,C,U]=hmflip(A,C,LC,LR,U,l,r)
    #Input:
    #A   - the cost matrix.
    #C   - the assignment vector.
    #LC  - the column label vector.
    #LR  - the row label vector.
    #U   - the 
    #r,l - position of last zero in path.
    #Output:
    #A   - updated cost matrix.
    #C   - updated assignment vector.
    #U   - updated unassigned row list vector.

    n = A.shape[0]

    while(1):
        #Move assignment in column l to row r.
        C[0,l-1] = r

        # Find zero to be removed from zero list..

        # Find zero before this.
        m = np.nonzero(A[r-1]==-l)[0] + 1

        # Link past this zero.
        A[r-1,m-1] = A[r-1,l-1]

        A[r-1,l-1] = 0
        
        # If this was the first zero of the path..
        if LR[0,r-1]<0:
            #...remove row from unassigned row list and return.
            U[0,n] = U[0,r-1]
            U[0,r-1] = 0
            return [A,C,U]
        else:

            # Move back in this row along the path and get column of next zero.
            l = LR[0,r-1]

            # Insert zero at (r,l) first in zero list.
            A[r-1,l-1] = A[r-1,n]
            A[r-1,n] = -l

            # Continue back along the column to get row of next zero in path.
            r = LC[0,l-1]
                
        
        
def hungarian(A):
    (m,n) = A.shape

    if (m!=n):
        print 'HUNGARIAN: Cost matrix must be square!'

    # Save original cost matrix.
    orig = A.copy()
                
    #Reduce matrix.
    A = hminired(A)

    # Do an initial assignment.
    [A,C,U] = hminiass(A)
    # Repeat while we have unassigned rows.
    while(U[0,n]!=0):
        #  Start with no path, no unchecked zeros, and no unexplored rows.
        LR = np.zeros((1,n))
        LC = np.zeros((1,n))
        CH = np.zeros((1,n))
        RH = np.hstack([np.zeros((1,n)), np.array([[-1]])])

        # No labelled columns.
        SLC = np.array([])

        # Start path in first unassigned row.
        r = U[0,n]
        # Mark row with end-of-path label.
        LR[0,r-1] = -1
        # Insert row first in labelled row set.
        SLR = np.array([r])
        # Repeat until we manage to find an assignable zero.
        while(1):
            # If there are free zeros in row r
            if A[r-1,n]!=0:
                # ...get column of first free zero.
                l = -A[r-1,n]

                # If there are more free zeros in row r and row r in not
                # yet marked as unexplored..
                if (np.logical_and(A[r-1,l-1]!=0,RH[0,r-1]==0)):
                    # Insert row r first in unexplored list.
                    RH[0,r-1] = RH[0,n]
                    RH[0,n] = r

                    # Mark in which column the next unexplored zero in this row is.
                    CH[0,r-1] = -A[r-1,l-1]

            else:
                # If all rows are explored..
                if RH[0,n] <= 0:
                    # Reduce matrix.
                    [A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR)

                # Re-start with first unexplored row.
                r = RH[0,n]
                # Get column of next free zero in row r.
                l = CH[0, r-1]
                # Advance "column of next free zero".
                CH[0, r-1] = -A[r-1,l-1]
                # If this zero is last in the list..
                if (A[r-1,l-1]==0):
                    # ...remove row r from unexplored list.
                    RH[0,n] = RH[0,r-1]
                    RH[0,r-1] = 0
            # While the column l is labelled, i.e. in path.
            while LC[0,l-1]!=0:
                # If row r is explored..
                if RH[0,r-1]==0:
                    # If all rows are explored..
                    if RH[0,n]<=0:
                        # Reduce cost matrix.
                        [A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR)
                        
                    # Re-start with first unexplored row.
                    r = RH[0,n]

                # Get column of next free zero in row r.
                l = CH[0,r-1]

                # Advance "column of next free zero".
                CH[0,r-1]=-A[r-1,l-1]

                # If this zero is last in list..
                if(A[r-1,l-1]==0):
                    # ...remove row r from unexplored list.
                    RH[0,n] = RH[0,r-1]
                    RH[0,r-1] = 0

            # If the column found is unassigned..
            if C[0,l-1] == 0:
                # Flip all zeros along the path in LR,LC.
                
                
                [A,C,U]=hmflip(A,C,LC,LR,U,l,r)
                # ...and exit to continue with next unassigned row.
                break
            else:
                #  ...else add zero to path.

                # Label column l with row r.
                LC[0,l-1] = r

                # Add l to the set of labelled columns.
                SLC = np.hstack([SLC, l])

                # Continue with the row assigned to column l.
                r = C[0,l-1]

                # Label row r with column l.
                LR[0,r-1]= l

                # Add r to the set of labelled rows.
                SLR = np.hstack([SLR, np.array([r])])
      #  Calculate the total cost.
    T = 0      
    for i in range(0,A.shape[0]):
	T = T + A[int(C[0][i])-1,i]
    return [C[0].astype(int),T]
                    
##A = loadmat('A.mat')['A']
##hungarian(A)
