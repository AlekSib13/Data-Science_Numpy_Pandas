#!/usr/bin/env python
# coding: utf-8

# ### Задание 1
# Импортируйте библиотеки pandas, numpy и matplotlib.
# Загрузите "Boston House Prices dataset" из встроенных наборов данных библиотеки sklearn.
# Создайте датафреймы X и y из этих данных.
# Разбейте эти датафреймы на тренировочные (X_train, y_train) и тестовые (X_test, y_test)
# с помощью функции train_test_split так, чтобы размер тестовой выборки
# составлял 20% от всех данных, при этом аргумент random_state должен быть равен 42.
# Масштабируйте данные с помощью StandardScaler.
# Постройте модель TSNE на тренировочный данных с параметрами:
# n_components=2, learning_rate=250, random_state=42.
# Постройте диаграмму рассеяния на этих данных.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='svg'")


# In[3]:


boston=load_boston()


# In[4]:


data=boston.data


# In[5]:


features=boston.feature_names
features


# In[6]:


X=pd.DataFrame(data,columns=features)
X


# In[7]:


y=pd.DataFrame(boston.target,columns=['price'])
y


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=42)


# In[10]:


import sklearn.preprocessing as pr


# In[11]:


my_scaler=pr.StandardScaler()


# In[12]:


X_scaled=my_scaler.fit_transform(X_train)


# In[13]:


X_train_scaled=pd.DataFrame(X_scaled,columns=features)


# In[14]:


X_train_scaled


# In[15]:


X_test_scaled=my_scaler.transform(X_test)


# In[16]:


X_test_scaled=pd.DataFrame(X_test_scaled,columns=features)


# In[17]:


X_test_scaled


# In[18]:


from sklearn.manifold import TSNE


# In[19]:


tsne=TSNE(n_components=2, learning_rate=250, random_state=42)


# In[20]:


X_train_scaled_tsne=tsne.fit_transform(X_train_scaled)


# In[21]:


X_train_scaled_tsne


# In[22]:


plt.scatter(X_train_scaled_tsne[:,0],X_train_scaled_tsne[:,1])


# In[23]:


#Вспомнить, как создать легенду, описание осей масштаб, цвет и т.д. для графиков: посмотреть видео


# ### Задание 2
# С помощью KMeans разбейте данные из тренировочного набора на 3 кластера,
# используйте все признаки из датафрейма X_train.
# Параметр max_iter должен быть равен 100, random_state сделайте равным 42.
# Постройте еще раз диаграмму рассеяния на данных, полученных с помощью TSNE,
# и раскрасьте точки из разных кластеров разными цветами.
# Вычислите средние значения price и CRIM в разных кластерах.

# In[24]:


from sklearn.cluster import KMeans


# In[25]:


model=KMeans(n_clusters=3,max_iter=100,random_state=42)


# In[26]:


train=model.fit_predict(X_train_scaled)


# In[27]:


pd.value_counts(train)


# In[28]:


plt.scatter(X_train_scaled_tsne[:,0],X_train_scaled_tsne[:,1],c=train)
plt.text(-25,-10,'Кластер 0')
plt.text(-0,-17,'Кластер 1')
plt.text(17,15,'Кластер 2')


# In[29]:


y_train.mean()


# In[30]:


y_train[train==0].mean()


# In[31]:


y_train[train==1].mean()


# In[32]:


y_train[train==2].mean()


# In[33]:


X_train['CRIM'][train==0].mean()


# In[34]:


X_train['CRIM'][train==1].mean()


# In[35]:


X_train['CRIM'][train==2].mean()

