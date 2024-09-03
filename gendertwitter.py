# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 15:30:26 2018

@author: Pc
"""

import pandas as pd

dataset =pd.read_csv(r'gendertwitter.csv',encoding='latin1')
dataset.head()

data =pd.concat([dataset.gender,dataset.description],axis=1)
data.dropna(inplace=True)
data.info()

data.gender = [1 if i == 'female' else 0 for i in data.gender]


#tidy data
import re 

# =============================================================================
# first=data.description[4]
# description= re.sub("[^a-zA-Z]"," ",first)
# 
# description =description.lower()
# 
# #irrelavant=stopworsds=anlamsız
# 
# import nltk as nlp
# from nltk.corpus import stopwords
# 
# description = nltk.word_tokenize(description) #bölme
# 
# description=[i for i in description if not i in set(stopwords.words("english"))] #grksizleri bulma
# 
# lema = nlp.WordNetLemmatizer()
# description = [ lema.lemmatize(j) for j in description]
# 
# =============================================================================
import nltk 
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lema = WordNetLemmatizer()

description_list = []
for i in data.description:
    description = re.sub("[^a-zA-Z]"," ",i)
    description = description.lower()
    description = nltk.word_tokenize(description) 
    #description=[j for j in i if not j in set(stopwords.words("english"))] #
    
    lema = nltk.WordNetLemmatizer()
    description= [ lema.lemmatize(j) for j in description]
    description = " ".join(description)
    description_list.append(description)



from sklearn.feature_extraction.text import CountVectorizer
max_feature=500

count_vectorizer= CountVectorizer(max_features=max_feature, stop_words ='english')

sparse_matrix = count_vectorizer.fit_transform(description_list).toarray()

print("ensık {} kelime {}".format(max_feature,count_vectorizer.get_feature_names()))



#machine learning
y = data.iloc[:,0].values
x = sparse_matrix

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x ,y , test_size= 0.1 ,random_state = 0)


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train ,y_train)

pred=nb.predict(x_test)
print("skor :",nb.score(pred.reshape(-1,1),y_test))











