import numpy as np 
from scipy.stats import bernoulli
from math import log,sqrt
Totalunits = [1000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000]

total_reward = np.zeros(len(Totalunits))
k =0;
for L in Totalunits:
	for x in range(0,100):	
		N=5
		# L=1000
		mu=0.1
		R=30
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
			return [alpha,beta]
		class Agents(object):
			"""docstring for Agents"""
			def __init__(self):
				super(Agents, self).__init__()
				self.Costs=np.random.rand(5)
				self.Capacities=np.ones(5)*L/N+4*L/N*np.random.rand(5)
				self.Qualities=np.array([0.85742449, 0.84950945, 0.67021904, 0.8082646 , 0.71201124])
			def getBids(self):
				bids=np.vstack((self.Costs,self.Capacities)).T
				return bids
			def reward(self,i):
				quality=self.Qualities[i]
				return get_ber_data(quality)
			def rewards(self):
				a=np.ones(N)
				for i in range(N):
					a[i]=self.reward(i)
				return a	
			def getCapacitiesFulfilled(self,numberOfTimesPlayed):
				return numberOfTimesPlayed<self.Capacities
			def compareBeta(self,beta):
				return self.Costs<beta
		a=Agents()
		bids=a.getBids()
		modifiedBids=[]
		for bid in bids:
			b=resampling(bid[0])
			modifiedBids.insert(len(modifiedBids),b)

		S=np.full(N,0.0)
		F=np.full(N,0.0)
		rewards=a.rewards()
		for i in range(N):
			if rewards[i]>0:
				S[i]+=1
			else:
				F[i]+=1
		# empericalQuality=rewards
		numberOfTimesPlayed=np.ones(N)
		# qualityUpperBound=empericalQuality+np.sqrt(log(N)/(2*numberOfTimesPlayed))
		t=N
		modifiedBids=np.array(modifiedBids)
		
		# print(modifiedBids)
		while t<L:
			#step7
			H=2*modifiedBids[:,0]
			#step8 p1
			temp=np.random.beta(S+1,F+1,N)
			temp2=1*a.getCapacitiesFulfilled(numberOfTimesPlayed)
			temp=temp*temp2
			i=np.argmax(temp)
			gi=1
			#step9
			if gi>0:
				#step 10,11
				reward=a.reward(i)
				total_reward[k]+= R*reward-H[i]
				numberOfTimesPlayed[i]+=1
				if reward>0:
					S[i]+=1
				else:
					F[i]+=1
			#step 12,13
			else:
				break
			t=t+1
		# print(bids[:,1])
		P=1/mu*numberOfTimesPlayed*(1-bids[:,0])
		temp=1*a.compareBeta(modifiedBids[:,1])
		P=P*temp
		T=bids[:,0]*numberOfTimesPlayed+P
		# print(T)
	print(total_reward[k]/(L*100))
	k = k+1

		# print(modifiedBids)
