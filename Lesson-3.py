#!/usr/bin/env python
# coding: utf-8

# ### Задание 1
# 
# Импортируйте библиотеки pandas и numpy.
# Загрузите "Boston House Prices dataset" из встроенных наборов данных библиотеки sklearn. Создайте датафреймы X и y из этих данных.
# Разбейте эти датафреймы на тренировочные (X_train, y_train) и тестовые (X_test, y_test) с помощью функции train_test_split так, чтобы размер тестовой выборки
# составлял 30% от всех данных, при этом аргумент random_state должен быть равен 42.
# Создайте модель линейной регрессии под названием lr с помощью класса LinearRegression из модуля sklearn.linear_model.
# Обучите модель на тренировочных данных (используйте все признаки) и сделайте предсказание на тестовых.
# Вычислите R2 полученных предказаний с помощью r2_score из модуля sklearn.metrics.
# 

# In[26]:


import numpy as np
import pandas as pd


# In[27]:


from sklearn.datasets import load_boston


# In[28]:


lbos=load_boston()


# In[29]:


lbos.keys()


# In[30]:


data=lbos.data


# In[31]:


target=lbos.target


# In[32]:


names=lbos.feature_names


# In[33]:


X=pd.DataFrame(data,columns=names)
X


# In[34]:


X.info()


# In[35]:


y=pd.DataFrame(target,columns=['price'])
y


# In[36]:


y.info()


# In[37]:


from sklearn.model_selection import train_test_split as tts


# In[39]:


X_train, X_test, y_train, y_test=tts(X,y,test_size=0.3,random_state=42)


# In[40]:


from sklearn.linear_model import LinearRegression


# In[41]:


lr=LinearRegression()


# In[42]:


lr.fit(X_train,y_train)


# In[43]:


y_pred=lr.predict(X_test)


# In[44]:


check=pd.DataFrame({'y_test':y_test['price'], 'y_pred':y_pred.flatten()},columns=['y_test','y_pred'])
check.head(20)


# In[45]:


from sklearn.metrics import r2_score


# In[46]:


r2_score(y_test,y_pred)


# ### Задание 2
# Создайте модель под названием model с помощью RandomForestRegressor из модуля sklearn.ensemble.
# Сделайте агрумент n_estimators равным 1000,
# max_depth должен быть равен 12 и random_state сделайте равным 42.
# Обучите модель на тренировочных данных аналогично тому, как вы обучали модель LinearRegression,
# но при этом в метод fit вместо датафрейма y_train поставьте y_train.values[:, 0],
# чтобы получить из датафрейма одномерный массив Numpy,
# так как для класса RandomForestRegressor в данном методе для аргумента y предпочтительно применение массивов вместо датафрейма.
# Сделайте предсказание на тестовых данных и посчитайте R2. Сравните с результатом из предыдущего задания.
# Напишите в комментариях к коду, какая модель в данном случае работает лучше.

# In[13]:


from sklearn.model_selection import GridSearchCV


# In[19]:


from sklearn.ensemble import RandomForestRegressor


# In[20]:


parameters=[{'n_estimators':[1000],'max_depth':[12]}]


# In[23]:


clf=GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                param_grid=parameters,
                scoring='accuracy',
                cv=5)


# In[52]:


clf.fit(X_train,y_train.values[:,0])
#далее решить задачу не удалось, поскольку после запуска данной строки появляется: 
# ValueError: continuous is not supported

