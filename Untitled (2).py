#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


books=pd.read_csv('Dataset/Books.csv',low_memory=False)
rating=pd.read_csv('Dataset/Ratings.csv',low_memory=False)
users=pd.read_csv('Dataset/Users.csv',low_memory=False)


# In[3]:


books


# In[47]:


# In[4]:


rating


# In[5]:


users


# In[6]:


print(books.shape)
print(users.shape)
print(rating.shape)


# In[7]:


books.isnull().sum()


# In[8]:


books.dropna(inplace=True)


# In[9]:


users.isnull().sum()


# In[10]:


rating.isnull().sum()


# In[11]:


books.duplicated().sum()


# # Popularity based recommender sysytem

# In[12]:


rating_with_name=rating.merge(books,on="ISBN")


# In[13]:


number_of_rating=rating_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
number_of_rating.rename(columns={'Book-Rating':'Number-of-rating'},inplace=True)


# In[14]:


number_of_rating


# In[15]:


avg=rating_with_name.groupby('Book-Title').mean()['Book-Rating'].reset_index()
avg.rename(columns={'Book-Rating':'Mean'},inplace=True)


# In[16]:


avg


# In[17]:


df1=avg.merge(number_of_rating,on='Book-Title')


# In[18]:


df1.head(50)


# In[19]:


df1=df1[df1['Number-of-rating']>=250].sort_values('Mean',ascending=False)
df1.shape


# In[20]:


df2=df1.merge(books,on='Book-Title').drop_duplicates('Book-Title')


# In[50]:


df2=df2[['Book-Title','Mean','Number-of-rating','Image-URL-L','Book-Author']]


# In[21]:


popular_df=df2.iloc[0:50]


# In[22]:


popular_df.shape


# #Collabrating filtering 

# In[23]:


x=rating_with_name.groupby('User-ID').count()
y=x[x['Book-Rating']>200].index


# In[24]:


y.shape


# In[25]:


filter_rating=rating_with_name[rating_with_name['User-ID'].isin(y)]


# In[26]:


filter_rating


# In[27]:


z=filter_rating.groupby('Book-Title').count()['Book-Rating']>=50
z=z[z].index


# In[28]:


z.shape


# In[29]:


final_data=filter_rating[filter_rating['Book-Title'].isin(z)]


# In[30]:


final_data.drop_duplicates()


# In[31]:


pv=final_data.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')
pv.shape


# In[32]:


pv.fillna(0,inplace=True)


# In[33]:


pv


# In[34]:


from sklearn.metrics.pairwise import cosine_similarity


# In[35]:


similarity_score=cosine_similarity(pv)


# In[36]:


def recommend(book_name):
    # index fetch
    index = np.where(pv.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1],reverse=True)[1:5]
    
    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pv.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-L'].values))
        
        data.append(item)
    
    return data


# In[37]:


sorted(list(enumerate(similarity_score[0])),key=lambda x:x[1],reverse=True)[1:6]


# In[38]:


recommend('Message in a Bottle')


# In[39]:


import pickle


# In[40]:





# In[41]:


pickle.dump(popular_df,open("popular.pkl",'wb'))
pickle.dump(pv,open('pv.pkl','wb'))
pickle.dump(books,open('books.pkl','wb'))
pickle.dump(similarity_score,open('similarity_scores.pkl','wb'))


# In[45]:





# In[46]:





# In[ ]:




