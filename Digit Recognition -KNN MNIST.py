#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[6]:


dfx=pd.read_csv('xdata.csv')
dfy=pd.read_csv('ydata.csv')


# In[9]:


x=dfx.values
y=dfy.values

x=x[:,1:] #Its for all row but droping the 1st column
y=y[:,1:]

print(x)
print(x.shape)
print(y.shape)


# In[12]:


plt.scatter(x[:,0],x[:,1],c=y)
plt.show()


# In[13]:


query_x=np.array([2,3])
plt.scatter(x[:,0],x[:,1],c=y)
plt.scatter(query_x[0],query_x[1],color='red')
plt.show()


# In[38]:


def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

def knn(X,Y,queryPoint,k=5):
    
    #Creating an empty array
    
    vals=[]
    
    m=X.shape[0]
    
    for i in range(m):
        d=dist(queryPoint,X[i])
        vals.append((d,Y[i])) #Append is used to add elements in an array
        
    vals=sorted(vals)
    vals=vals[:k]
        
    vals=np.array(vals)
    
    #print(vals)
    
    new_vals=np.unique(vals[:,1],return_counts=True)
    print(new_vals)
    
    index=new_vals[1].argmax()
    pred=new_vals[0][index]
    
    return pred


# In[39]:


knn(x,y,query_x)


# #MNIST Dataset

# In[35]:


df=pd.read_csv('train.csv')
print(df.shape)


# In[36]:


print(df.columns)


# In[37]:


df.head()


# #Create Numpy Array

# In[40]:


data=df.values
print(data.shape)
print(type(data))


# In[46]:


x=data[:,1:]
y=data[:,0]

print(x.shape,y.shape)


# In[49]:


split=int(0.8*x.shape[0])
print(split)


# In[50]:


x_train=x[:split,:]
y_train=y[:split]
x_test=x[split:,:]
y_test=y[split:]

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


# In[53]:


def drawImg(sample):
    img=sample.reshape((28,28))
    plt.imshow(img)
    plt.show()


# In[54]:


drawImg(x_train[3])
print(y_train[3])


# In[55]:


def drawImg(sample):
    img=sample.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.show()
    
drawImg(x_train[3])
print(y_train[3])


# In[56]:


pred=knn(x_train,y_train,x_test[0])
print(pred)


# In[57]:


drawImg(x_test[0])
print(y_test[0])


# In[58]:


drawImg(x_test[7])
print(y_test[7])


# In[ ]:





# In[ ]:




