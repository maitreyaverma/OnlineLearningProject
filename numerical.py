import numpy as np 
from scipy.stats import bernoulli
from math import log,sqrt
mu=0.5
def get_ber_data(p):
	result = bernoulli.rvs(p,size=1)
	return result[0]
def recursive(c):
	a=get_ber_data(1-mu)
	if a==1:
		return c
	else:
		temp=np.random.rand(1)[0]*(1-c)+c
		return recursive(temp)
def resampling(c):
	a=get_ber_data(1-mu)
	alpha=0.0
	beta=0.0
	if a==1:
		alpha=c
		beta=c
	else:
		beta=np.random.rand(1)[0]*(1-c)+c
		alpha=recursive(beta)
	# return [alpha,beta]
	return alpha
for c in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
	mysum=0
	N=10000
	for i in range(N):
		mysum+=resampling(c)
	print(mysum/N)