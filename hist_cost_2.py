import numpy as np



def hist_cost_2(BH1,BH2, r_inner, r_outer, nbins_theta, nbins_r):

	#BH1 and BH2 are 2-D matrices. r_inner, r_outer, nbins_theta, nbins_r are not used in this program. final output is matrix 
	#HC of size nsamp1*nsamp2

	(nsamp1, nbins) = BH1.shape

	(nsamp2, nbins) = BH2.shape

	#(i,j)th entry of BH1n = (i,j)th entry of BH1/colsum(i). Same for BH2. Can be interpreted as a normalization for each column.

	BH1n = np.zeros((nsamp1, nbins))			#initializing BH1n

	col_sum = np.sum(BH1, 1)			#computing column sum

	col_sum = col_sum + np.spacing(1)		#adding a very small value to every column sum

	for i in range(nsamp1):

		for j in range(nbins):

			BH1n[i,j] = BH1[i,j]/col_sum[i]		#populating BH1n




	BH2n = np.zeros((nsamp2, nbins))			

	col_sum = np.sum(BH2, 1)

	col_sum = col_sum + np.spacing(1)		

	for i in range(nsamp2):

		for j in range(nbins):

			BH2n[i,j] = BH2[i,j]/col_sum[i]	


	
	#We perform the following steps: (i) For every i, extract column i = C_{i} from BH1n. (ii) M_{i} = [C_{i}...C_{i}] 

	#nsamp2 times.	These M_{i}'s are stored in a list called tmp1 (not the same as tmp1 of MATLAB code)


	tmp1 = []

	for i in range(nbins):

		extracted_col = BH1n[:,i]					#this comes out as a row vector. 

		extracted_col = np.tile(extracted_col,(1,1)).transpose( )	#We now have the ith column

		M = np.tile(extracted_col,(1, nsamp2))

		tmp1.append(M)


	#For BH2n, we do the following: (i) For every i extract column i = C_{i} from BH2n. (ii) M_{i} = [C_{i}'...C_{i}']

	#nsamp1 times (' denotes transpose, MATLAB notation). These M_{i}'s are stored in a list called tmp2.


	tmp2 = []

	for i in range(nbins):

		extracted_col = BH2n[:,i]					#this comes out as a row vector

		M = np.tile(extracted_col,(nsamp1,1))

		tmp2.append(M)

	
	#For every i, tmp1[i] and tmp2[i] are matries of size nsamp1*nsamp2. Compute element-wise difference of tmp1[i] and tmp2[i],  
	
	#square it and divide this entry by sum of corresponding entries in tmp1[i] and tmp2[i]. Store all these matrices in 

	#tmp1 itself.

	for i in range(nbins):

		M = (tmp1[i] - tmp2[i])**2/(tmp1[i] + tmp2[i] + np.spacing(1))

		tmp1[i] = M


	#Now perform element-wise addition of all matrices in tmp1 to get the final o/p HC. It is a matrix of size nsamp1*nsamp2.


	for i in range(1, nbins):

		tmp1[0] = tmp1[0] + tmp1[i]


	HC = tmp1[0]

	return HC



		

		

		


	

	

	



	

	

