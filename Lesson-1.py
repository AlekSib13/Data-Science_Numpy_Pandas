#!/usr/bin/env python
# coding: utf-8

# # Numpy.Задание 1
# Импортируйте библиотеку Numpy и дайте ей псевдоним np.
# Создайте массив Numpy под названием a размером 5x2, то есть состоящий из 5 строк и 2 столбцов. 
# Первый столбец должен содержать числа 1, 2, 3, 3, 1, а второй - числа 6, 8, 11, 10, 7. Будем считать, 
# что каждый столбец - это признак, а строка - наблюдение. Затем найдите среднее значение по каждому признаку, 
# используя метод mean массива Numpy. Результат запишите в массив mean_a, в нем должно быть 2 элемента.

# In[2]:


import numpy as np


# In[6]:


a=np.array([[1,6],
            [2,8],
            [3,11],
            [3,10],
            [1,7]])
a


# In[7]:


np.mean(a,axis=0)


# In[40]:


mean_a=np.array(np.mean(a,axis=0))
mean_a


# ## Numpy.Задание 2
# Вычислите массив a_centered, отняв от значений массива “а” средние значения соответствующих признаков, 
# содержащиеся в массиве mean_a. Вычисление должно производиться в одно действие. Получившийся массив должен иметь размер 5x2.

# In[9]:


a_centered=np.subtract(a,mean_a)
a_centered


# ## Numpy.Задание 3
# Найдите скалярное произведение столбцов массива a_centered. 
# В результате должна получиться величина a_centered_sp. Затем поделите a_centered_sp на N-1, где N - число наблюдений.

# In[24]:


c_cen=np.transpose(a_centered)
a_centered_sp=np.dot(c_cen[0],c_cen[1])
a_centered_sp


# In[27]:


s=np.sum(a)
result=a_centered_sp/(s-1)
result


# In[37]:


a[2,1]


# ## Pandas. Задание 1
# Импортируйте библиотеку Pandas и дайте ей псевдоним pd. Создайте датафрейм authors со столбцами author_id и author_name, в которых соответственно содержатся данные: [1, 2, 3] и ['Тургенев', 'Чехов', 'Островский'].
# Затем создайте датафрейм book cо столбцами author_id, book_title и price, в которых соответственно содержатся данные: 
# [1, 1, 1, 2, 2, 3, 3], 
# ['Отцы и дети', 'Рудин', 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой', 'Гроза', 'Таланты и поклонники'],
# [450, 300, 350, 500, 450, 370, 290].
# 

# In[41]:


import pandas as pd


# In[43]:


authors=pd.DataFrame({'author_id':[1,2,3],'author_name':['Тургенев', 'Чехов', 'Островский']}, 
                     columns=['author_id', 'author_name'])
authors


# In[45]:


book=pd.DataFrame({'author_id':[1, 1, 1, 2, 2, 3, 3], 
                   'book_title':['Отцы и дети', 'Рудин', 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой', 'Гроза', 'Таланты и поклонники'],
                  'price':[450, 300, 350, 500, 450, 370, 290]}, columns=['author_id','book_title','price'])
book


# ## Pandas. Задание 2
# Получите датафрейм authors_price, соединив датафреймы authors и books по полю author_id.
# 

# In[54]:


authors_price=pd.merge(authors, book, on='author_id',how='outer')
authors_price


# ## Pandas. Задание 3
# Создайте датафрейм top5, в котором содержатся строки из authors_price с пятью самыми дорогими книгами.
# 

# In[74]:


top5=authors_price.groupby('book_title').agg({'price':'max'})
top5.nlargest(5,'price')


# ## Задание 4
# Создайте датафрейм authors_stat на основе информации из authors_price. В датафрейме authors_stat должны быть четыре столбца:
# author_name, min_price, max_price и mean_price,
# в которых должны содержаться соответственно имя автора,минимальная, максимальная и средняя цена на книги этого автора.b

# In[94]:


author_mp=authors_price.groupby('author_name').agg({'price':'min'})
author_mp=author_mp.rename(columns={'price':'min_price'})
author_mp


# In[92]:


author_mx=authors_price.groupby('author_name').agg({'price':'max'})
author_mx=author_mx.rename(columns={'price':'max_price'})
author_mx


# In[90]:


author_meanp=authors_price.groupby('author_name').agg({'price':'mean'})
author_meanp=author_meanp.rename(columns={'price':'mean_price'})
author_meanp


# In[95]:


authors_stat=pd.concat([author_mp,author_mx,author_meanp],axis=1)
authors_stat

