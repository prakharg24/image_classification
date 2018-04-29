
# coding: utf-8

# In[3]:


import numpy as np
import os
from sklearn.cluster import KMeans
import csv
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm


# In[4]:


def remnpy(filename):
	for i in range(len(filename)):
		if(filename[i:i+4]==".npy"):
			return filename[:i]
	return ""


# In[32]:


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


# In[33]:


inp = StandardScaler().fit_transform(inp)


# In[34]:


pca = PCA(n_components=10)
inputs_pca = pca.fit_transform(inp)


# In[35]:


test_inp = np.load("test/test.npy")
test_inp = StandardScaler().fit_transform(test_inp)
test_pca = pca.transform(test_inp)


# In[36]:


print(np.shape(inp_tag))
print(np.shape(inputs_pca))
inp_tag = np.reshape(inp_tag, [-1, 1])
new_arr = np.concatenate((inputs_pca, inp_tag), axis=1)
np.random.shuffle(new_arr)
inputs_pca = new_arr[:, :10]
inp_tag = new_arr[:, 10:]
inp_tag = np.reshape(inp_tag, [-1])


# In[ ]:


print(np.shape(inputs_pca))
avg_ttl = 0.0
for i in range(0, 10):
    svm_model = svm.SVC(C=1, decision_function_shape='ovo')
    svm_model.fit(np.concatenate((inputs_pca[:10000*i], inputs_pca[10000*(i+1):])), np.concatenate((inp_tag[:10000*i], inp_tag[10000*(i+1):])))
    all_outs = svm_model.predict(inputs_pca[10000*i:10000*(i+1)])
    count = 0
    total = 0
    for ele, real in zip(all_outs, inp_tag[10000*i:10000*(i+1)]):
        if(ele==real):
            count += 1
        total += 1

    avg_ttl += (count + 0.0)/total
    
print(avg_ttl/10)


# In[18]:


myFile = open('pca.csv', 'w', newline='')
with myFile:
    fieldnames = ['ID', 'CATEGORY']
    writer = csv.DictWriter(myFile, fieldnames=fieldnames)
    writer.writeheader()
    for i, ele in enumerate(all_outs):
        writer.writerow({'ID' : str(i), 'CATEGORY' : inp_dict[ele]})

inp_acc = svm_model.predict(inputs_pca)

count = 0
total = 0
for ele, real in zip(inp_acc, inp_tag):
    if(ele==real):
        count += 1
    total += 1

print(count)
print(total)
print((count + 0.0)/total)

