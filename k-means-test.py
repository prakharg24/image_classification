
# coding: utf-8

# In[ ]:


import numpy as np
import os
from sklearn.cluster import KMeans
import csv
import pickle


# In[ ]:


with open('kmeans_data.pkl', 'rb') as f:
    kmeans = pickle.load(f)
with open('kmeans_tags.pkl', 'rb') as f:
    inp_tag = pickle.load(f)


# In[ ]:


tlen = 20
max_count = np.zeros([tlen, tlen])

for i, ele in enumerate(kmeans.labels_):
	max_count[ele][inp_tag[i]] += 1

wrong = 0
total = 0


# print(max_count)
max_ind = np.argmax(max_count, axis=1)
print(max_ind)

for i in range(0, tlen):
	for j in range(0, tlen):
		if(j!=max_ind[i]):
			wrong += max_count[i][j]
# 			print(max_count[i][j])
		total += max_count[i][j]

print((total - wrong + 0.0)/total)


# In[ ]:


test_inp = np.load("test/test.npy")
new_arr = kmeans.predict(test_inp)
ans_arr = []
for ele in new_arr:
	ans_arr.append(max_ind[ele])


# In[ ]:


myFile = open('ans.csv', 'w', newline='')
with myFile:
    fieldnames = ['ID', 'CATEGORY']
    writer = csv.DictWriter(myFile, fieldnames=fieldnames)
    writer.writeheader()
    for i, ele in enumerate(ans_arr):
        writer.writerow({'ID' : str(i), 'CATEGORY' : inp_dict[ele]})

