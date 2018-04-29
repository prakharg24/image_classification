
# coding: utf-8

# In[93]:


import numpy as np
import os
from sklearn.cluster import KMeans
import csv
import pickle


# In[83]:


def remnpy(filename):
	for i in range(len(filename)):
		if(filename[i:i+4]==".npy"):
			return filename[:i]
	return ""


# In[123]:


files = os.listdir("train")
inp = np.array([])
inp = np.reshape(inp, [-1, 784])
inp_tag = []
inp_dict = {}
ite = 0
for file in files:
    inp = np.concatenate((inp, np.load("train/" + file)), axis = 0)
    inp_dict[ite] = remnpy(file)
    for i in range(0, 5000):
    	inp_tag.append(ite)
    ite += 1
inp_tag = np.array(inp_tag)


# In[151]:


tlen=20
kmeans = KMeans(n_clusters=tlen)
kmeans.fit(inp)
with open('kmeans_data.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
with open('kmeans_tags.pkl', 'wb') as f:
    pickle.dump(inp_tag, f)

