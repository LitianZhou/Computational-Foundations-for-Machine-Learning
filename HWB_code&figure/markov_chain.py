# no imports beyond numpy should be used in answering this question
import numpy as np

p_x1 = np.array([0.1, 0.3, 0.6])
p_x2_given_x1 = p_x3_given_x2 = p_x4_given_x3 = \
    np.array([[0.3, 0.5, 0.2],
    [0.37, 0.33, 0.3],
    [0.3, 0.5, 0.2]])

def calculate_probability(x1=0, x2=0, x3=0, x4=0):
	"""
	Calculate P(x1,x2,x3,x4). Each of these can take 3 values {1,2,3}. 
	If the value of any random variable input to this function is 0, marginalize out the random variable.
	"""

	probability = 1.0
	
	# Do not edit any code outside the edit region
	# Edit region starts here
	#########################
	# Your code goes here
	if 'p_table' not in calculate_probability.__dict__:
		p_table = -np.ones([4,4,4,4])
	if p_table[x1,x2,x3,x4] != -1:
		return p_table[x1,x2,x3,x4]
	para = [x1,x2,x3,x4]
	if 0 not in para:
		p_table[x1,x2,x3,x4] = p_x1[x1-1]*p_x2_given_x1[x1-1, x2-1]*p_x3_given_x2[x2-1,x3-1]*p_x4_given_x3[x3-1, x4-1]
		return p_table[x1,x2,x3,x4]
	else:
		for i,x in enumerate(para):
			if x==0:
				temp_para = para[:]
				temp_p = 0.0
				for j in range(1,4):
					temp_para[i] = j
					temp_p += calculate_probability(temp_para[0],temp_para[1],temp_para[2],temp_para[3])
				p_table[x1,x2,x3,x4] = temp_p
				return temp_p
	#########################
	# Edit region ends here

	return probability

if __name__ == '__main__':
	# Q1 = P(X_1=1,X_2=3,X_3=2,X_4=1)
	result1 = calculate_probability(x1=1,x2=3,x3=2,x4=1)
	print('P(X_1=1,X_2=3,X_3=2,X_4=1) =', result1)

	# Q2 = P(X_1=1,X_2=3,X_3=2,X_4=2)
	result2 = calculate_probability(x1=1,x2=3,x3=2,x4=2)
	print('P(X_1=1,X_2=3,X_3=2,X_4=2) =', result2)

	# Q3 = P(X_1=1,X_2=3,X_3=2)
	result3 = calculate_probability(x1=1,x2=3,x3=2,x4=0)
	print('P(X_1=1,X_2=3,X_3=2) =', result3)

	# Q4 = P(X_1=1,X_3=2)
	result4 = calculate_probability(x1=1,x2=0,x3=2,x4=0)
	print('P(X_1=1,X_3=2) =', result4)

	# Q5 = P(X_3=2)
	result5 = calculate_probability(x1=0,x2=0,x3=2,x4=0)
	print('P(X_3=2) =', result5)
