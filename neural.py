import re
import unicodedata
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pickle
import os
import numpy as np
from sklearn.cluster import KMeans
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm

class Model(torch.nn.Module) :
	def __init__(self,input_dim,hidden_dim) :
		super(Model,self).__init__()
		self.hidden_dim = hidden_dim
		self.linearOut1 = nn.Linear(input_dim, hidden_dim)
		self.linearOut2 = nn.Linear(hidden_dim, 20)
	def forward(self,inputs):
		x = self.linearOut1(inputs)
		x = F.sigmoid(x)
		x = self.linearOut2(x)
		x = F.log_softmax(x, dim=1)
		return x

def remnpy(filename):
	for i in range(len(filename)):
		if(filename[i:i+4]==".npy"):
			return filename[:i]
	return ""

files = os.listdir("/home/cse/btech/cs1150245/scratch/train")
inp = np.array([])
inp = np.reshape(inp, [-1, 784])
inp_tag = []
inp_dict = {}
ite = 0
for file in files:
    inp = np.concatenate((inp, np.load("/home/cse/btech/cs1150245/scratch/train/" + file)), axis = 0)
    inp_dict[ite] = remnpy(file)
    for i in range(0, 5000):
    	inp_tag.append(ite)
    ite += 1

inp_tag = np.array(inp_tag)

inp_tag = np.reshape(inp_tag, [-1, 1])
new_arr = np.concatenate((inp, inp_tag), axis=1)
np.random.shuffle(new_arr)
inp = new_arr[:, :784]
inp_tag = new_arr[:, 784:]
inp_tag = np.reshape(inp_tag, [-1])

hd = 1000
model = Model(784, hd)

error = 0
try:
	model.load_state_dict(torch.load('/home/cse/btech/cs1150245/scratch/model' + str(hd) + '.pth'))
except:
	error = 1

loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 0


acc_inp = Variable(torch.FloatTensor(inp))
y_pred = model(acc_inp)
lm = y_pred.data.cpu().numpy()
t = np.argmax(lm, axis = 1)
corr =0
total =0
for ele, pred in zip(t, inp_tag):
	if(ele==pred):
		corr += 1
	total += 1

print("Accuracies :")
print(corr)
print(total)
print((corr + 0.0)/total)

avg_ttl = 0.0

for counter in range(0, 20):
	for i in range(epochs) :
		avg_loss = 0.0
		for idx in range(0, 20):
			if(idx==counter):
				# continue
			input_data = Variable(torch.FloatTensor(inp[idx*5000: (idx+1)*5000]))
			target_data = Variable(torch.LongTensor(inp_tag[idx*5000: (idx+1)*5000]))
			y_pred = model(input_data)
			model.zero_grad()
			loss = loss_function(y_pred, target_data)
			avg_loss += loss.data[0]
			# print('epoch :%d iterations :%d loss :%g'%(i, idx, loss.data[0]))
			loss.backward()
			optimizer.step()

		if((i+1)%10==0):
			torch.save(model.state_dict(), '/home/cse/btech/cs1150245/scratch/model' + str(hd) + '.pth')
			print('the average loss after completion of %d epochs is %g'%((i+1),(avg_loss/20)))


# print(avg_ttl/10)
# test_inp = np.load("/home/cse/btech/cs1150245/scratch/test/test.npy")
# acc_inp = Variable(torch.FloatTensor(test_inp))
# y_pred = model(acc_inp)
# lm = y_pred.data.cpu().numpy()
# ans_arr = np.argmax(lm, axis = 1)

# myFile = open('/home/cse/btech/cs1150245/scratch/ans.csv', 'w', newline='')
# with myFile:
#     fieldnames = ['ID', 'CATEGORY']
#     writer = csv.DictWriter(myFile, fieldnames=fieldnames)
#     writer.writeheader()
#     for i, ele in enumerate(ans_arr):
#         writer.writerow({'ID' : str(i), 'CATEGORY' : inp_dict[ele]})