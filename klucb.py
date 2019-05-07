import numpy as np 
from scipy.stats import bernoulli
from math import log,sqrt
Totalunits = [1000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000]
Totalunits = [1000]
total_reward = np.zeros(len(Totalunits))
k =0;
for L in Totalunits:
	for x in range(0,1):	
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

		rewards=a.rewards()
		empericalQuality=rewards
		numberOfTimesPlayed=np.ones(N)
		qualityUpperBound=empericalQuality+np.sqrt(log(N)/(2*numberOfTimesPlayed))

		t=N
		modifiedBids=np.array(modifiedBids)
		
		# print(modifiedBids)
		while t<L:
			#step7
			H=2*modifiedBids[:,0]
			#step8 p1
			temp=np.ones(N)
			for i in range(N):
				armMean=R*qualityUpperBound[i]-H[i]
				required_value=(log(t)+3*log(log(t)))/(armMean+0.01)
				left_lim=armMean
				right_lim=1.0
				pa=armMean
				mid=0.0
				while right_lim - left_lim >0.001:
					# print(left_lim)
					mid=(left_lim+right_lim)/2
					if pa==0:
						if mid==1:
							kl_div=50000
							break
						else:
							kl_div=(1- pa)*log(( 1 -pa)/( 1 - mid))
					elif pa==1:
						kl_div=pa*log(pa/mid)
					else:
						kl_div=pa*log(pa/mid) + (1-pa)*log((1-pa)/(1-mid))
					if kl_div < required_value:
						left_lim=mid
					else:
						right_lim=mid
				temp[i]=mid
			temp2=1*a.getCapacitiesFulfilled(numberOfTimesPlayed)
			temp=temp*temp2
			i=np.argmax(temp)
			gi=R*qualityUpperBound[i]-H[i]
			#step9
			if gi>0:
				#step 10,11
				reward=a.reward(i)
				total_reward[k]+= R*reward-H[i]
				empericalQuality[i]=(empericalQuality[i]*numberOfTimesPlayed[i]+reward)/(numberOfTimesPlayed[i]+1)
				numberOfTimesPlayed[i]+=1
				qualityUpperBound[i]=empericalQuality[i]+sqrt(log(t)/(2*numberOfTimesPlayed[i]))
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
	print(total_reward[k]/(L))
	k = k+1

		# print(modifiedBids)
