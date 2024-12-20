#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import string
import spacy
from spacy import displacy
import nltk
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, SpatialDropout1D, Flatten, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPool1D, Dense
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from keras.models import load_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix, classification_report


# In[4]:


get_ipython().system('pip install keras')
get_ipython().system('pip install tensorflow')


# In[10]:


nlp = spacy.load('en_core_web_lg')


# In[9]:


get_ipython().system('python -m spacy download en_core_web_lg')


# In[11]:


nltk.download('stopwords')


# In[ ]:


"C:\Users\selvi\Downloads\hotel_reviews.xlsx"


# In[15]:


hotel = pd.read_excel("C:/Users/selvi/Downloads/hotel_reviews.xlsx")


# In[16]:


hotel.head()


# In[18]:


hotel.info()


# # NULL CHECK

# In[19]:


hotel.isna().sum()


# #spaces Check

# In[20]:


blanks = []

for i, rv, rt in hotel.itertuples():
    if rv.isspace():
        blanks.append(i)


# In[21]:


blanks


# In[22]:


hotel['Rating'].value_counts()


# # categorical conversion

# In[23]:


hotel['Rating'] = pd.Categorical(hotel['Rating'])


# In[24]:


hotel.info()


# # visualization POS

# In[25]:


displacy.render(nlp(hotel['Review'][1]), style='ent', jupyter=True)


# In[26]:


def word_cleaner(text):
    
    text = text.strip()
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    doc = nlp(text)

    lemmatized_words = [token.lemma_ for token in doc]
    
    additional_stopwords = set(['hotel', 'resort', 'day', 'use', 'need', 'think', 'night', 'say', 'look', 'beach', 'stay', 'time', 'people', 'place', 'area', 'room', 'come', 'staff', 'tell'])

    stop_words = set(stopwords.words('english')).union(set(spacy.lang.en.stop_words.STOP_WORDS)).union(additional_stopwords) - {'not'}
    
    lemmatized_words = [word for word in lemmatized_words if word.lower() not in stop_words]

    cleaned_text = ' '.join(lemmatized_words)

    return cleaned_text


# Executed the below code cell, Since SpaCy was used for lemmatization it was taking time and hence dumped the dataframe to another excel 

# In[31]:


hotel['cleaned_reviews'] = hotel['Review'].apply(word_cleaner)
hotel.to_excel('cleaned_hotel.xlsx')


# In[30]:


hotel = pd.read_excel("C:/Users/selvi/Downloads/cleaned_hotel.xlsx")


# In[32]:


displacy.render(nlp(hotel['cleaned_reviews'][1]), style='ent', jupyter=True)


# In[33]:


displacy.render(nlp(hotel['cleaned_reviews'][1]), style='dep', jupyter=True)


# In[34]:


vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(hotel['Review'])
word_count = pd.DataFrame(X.sum(axis=0), columns=vectorizer.get_feature_names_out()).T.sort_values(0, ascending=False)
word_count.columns = ['count']

plt.figure(figsize=(12, 6))
sns.barplot(x='count', y=word_count.index[:10], data=word_count[:10], palette='viridis')
plt.title('Top 10 Most Frequent Words')
plt.xlabel('Word Count')
plt.ylabel('Words')
plt.show()


# In[35]:


sns.countplot(x='Rating', data=hotel)
plt.title('Distribution of Ratings')
plt.show()


# In[36]:


hotel['Review_Length'] = hotel['Review'].apply(len)
sns.histplot(hotel['Review_Length'], bins=30, kde=True)
plt.title('Distribution of Review Lengths')
plt.xlabel('Review Length')
plt.ylabel('Frequency')
plt.show()


# In[37]:


max_seq_len = hotel['Review_Length'].max()


# In[40]:


sns.pairplot(hotel, vars=['Rating', 'Review_Length'], hue='Rating', palette='viridis')
plt.suptitle('Pairplot of Ratings and Review Lengths', y=1.02)
plt.show()


# # Sentiment Analysis - Polarity Scores

# In[42]:


sid = SentimentIntensityAnalyzer()
def polarity(text):
    return TextBlob(text).sentiment.polarity


# In[41]:


import nltk
nltk.download('vader_lexicon')


# In[43]:


hotel['Polarity_Scores'] = hotel['cleaned_reviews'].apply(lambda review: sid.polarity_scores(review)['compound'])


# In[44]:


hotel.head(15)


# In[45]:


hotel[hotel['Rating'] < 3].iloc[0,1:3]


# In[46]:


all_text = ' '.join(hotel['Review'])
wordcloud = WordCloud(max_words=500, width=800, height=400, random_state=42, max_font_size=110).generate(all_text)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Word Cloud of Text')
plt.show()


# In[47]:


all_text = ' '.join(hotel['cleaned_reviews'])
wordcloud = WordCloud(max_words=500, width=800, height=400, random_state=42, max_font_size=110).generate(all_text)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Word Cloud of Cleaned Text')
plt.show()


# In[48]:


hotel['Polarity_Scores'].max()


# In[49]:


hotel['Polarity_Scores'].min()


# In[50]:


def sentiment(label):
    if label < 0:
        return "Negative"
    elif label >= 0:
        return "Positive"


# In[51]:


hotel['Sentiment'] = hotel['Polarity_Scores'].apply(sentiment)


# In[52]:


hotel.head()


# In[53]:


hotel.drop(columns=['Unnamed: 0', 'Review', 'Review_Length'], inplace=True)


# In[54]:


hotel.head()


# In[55]:


hotel['Rating'][hotel['Polarity_Scores'] > 0].value_counts()


# In[56]:


hotel['Rating'][hotel['Polarity_Scores'] < 0].value_counts()


# In[57]:


pos_revs = hotel[hotel.Sentiment == 'Positive']
pos_revs = pos_revs.sort_values(['Polarity_Scores'], ascending=False)
pos_revs.head()


# In[58]:


text = ' '.join([word for word in pos_revs['cleaned_reviews']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500,width=1600,height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in positive reviews')
plt.show()


# In[59]:


neg_revs = hotel[hotel.Sentiment == 'Negative']
neg_revs = neg_revs.sort_values(['Polarity_Scores'], ascending=False)
neg_revs.head()

text = ' '.join([word for word in neg_revs['cleaned_reviews']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500,width=1600,height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in negative reviews')
plt.show()


# # Vectorization

# In[60]:


X = hotel['cleaned_reviews']
y = hotel['Sentiment']

vectorizer = TfidfVectorizer()

X_vector = vectorizer.fit_transform(X)

oversample = RandomOverSampler(sampling_strategy='minority')
X_over, y_over = oversample.fit_resample(X_vector, y)
undersample = RandomUnderSampler(sampling_strategy='majority')
X_combined, y_combined = undersample.fit_resample(X_over, y_over)

X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.3, random_state=12)


# # Logistic Regreesion

# In[61]:


logreg_model = LogisticRegression(random_state=42)
logreg_model.fit(X_train, y_train)


# In[62]:


predictions = logreg_model.predict(X_test)


# In[63]:


logreg_model.score(X_test , y_test)


# In[64]:


print(classification_report(y_test, predictions))


# In[65]:


confusion_matrix(y_test, predictions)


# In[66]:


lg_ac = accuracy_score(y_test, predictions)


# # SVC

# In[68]:


clf_svm = SVC(kernel='linear', random_state=42)

clf_svm.fit(X_train, y_train)


# In[69]:


clf_svm.score(X_test , y_test)


# In[70]:


predictions = clf_svm.predict(X_test)


# In[71]:


print(classification_report(y_test, predictions))


# In[72]:


confusion_matrix(y_test, predictions)


# In[73]:


sv_ac = accuracy_score(y_test, predictions)


# In[74]:


clf_svm.predict(vectorizer.transform(["Hotel is bad, but people are very good and service is good"]))


# # Random Forest

# In[75]:


clf_rf = RandomForestClassifier(random_state=42, max_depth=10)


# In[76]:


clf_rf.fit(X_train, y_train)


# In[77]:


rf_score = clf_rf.score(X_test, y_test)
print("Random Forest Classifier Score:", rf_score)


# In[78]:


predictions_rf = clf_rf.predict(X_test)


# In[79]:


print(classification_report(y_test, predictions_rf))


# In[80]:


print(confusion_matrix(y_test, predictions_rf))


# In[81]:


rf_ac=accuracy_score(y_test, predictions_rf)
rf_ac


# In[82]:


rf_cv_scores = cross_val_score(clf_rf, X_train, y_train, cv=5)


# # CNN

# In[83]:


max_words = 2500
embed_dim = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(hotel.cleaned_reviews.values)
word_index = tokenizer.word_index
X_cnn = tokenizer.texts_to_sequences(hotel.cleaned_reviews.values)


# In[84]:


X_cnn = pad_sequences(X_cnn, maxlen=max_seq_len)
y_cnn = pd.get_dummies(hotel['Sentiment']).values

X_cnn_train, X_cnn_test, y_cnn_train, y_cnn_test = train_test_split(X_cnn, y_cnn, test_size=0.2, random_state=42)


# In[85]:


X_cnn_train.shape


# In[86]:


y_cnn_train.shape


# In[ ]:





# In[ ]:





# In[108]:


cnn = Sequential()
cnn.add(Embedding(max_words, embed_dim, input_length=X_cnn_train.shape[1]))
cnn.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
cnn.add(GlobalMaxPool1D())
cnn.add(Dense(64, activation='relu'))
cnn.add(Dense(2, activation='softmax'))
cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = cnn.fit(X_cnn_train, y_cnn_train, epochs=5, batch_size=64, validation_split=0.1)


# In[109]:


cnn.save("trained_cnn_model.h5")


# In[110]:


cnn = load_model("trained_cnn_model.h5")


# In[111]:


cnn_ac = cnn.evaluate(X_cnn_test, y_cnn_test)
print(f'Loss: {cnn_ac[0]:.3f}\nAccuracy: {cnn_ac[1]:.3f}')


# # Decision Tree - Gini

# In[88]:


cart_gini = DecisionTreeClassifier(criterion='gini', max_depth=39)


# In[89]:


cart_gini_cv_scores = cross_val_score(cart_gini, X_train, y_train, cv=5)


# In[90]:


cart_gini.fit(X_train, y_train)


# In[91]:


pred_gini = cart_gini.predict(X_test)
cartg_ac = accuracy_score(y_test, pred_gini)
print("Decision Tree Classifier Accuracy:", cartg_ac)


# In[92]:


print(classification_report(y_test, pred_gini))


# In[93]:


print("Cross-Validation Scores:", cart_gini_cv_scores)
print("Mean Cross-Validation Score:", np.mean(cart_gini_cv_scores))


# # Decision Tree - Entropy

# In[94]:


cart_ent = DecisionTreeClassifier(criterion='entropy', max_depth=39)


# In[95]:


cart_ent_cv_scores = cross_val_score(cart_ent, X_train, y_train, cv=5)


# In[96]:


cart_ent.fit(X_train, y_train)


# In[97]:


pred_ent = cart_ent.predict(X_test)
carte_ac = accuracy_score(y_test, pred_ent)
print("Decision Tree Classifier Accuracy:", carte_ac)


# In[98]:


print(classification_report(y_test, pred_ent))


# In[99]:


print("Cross-Validation Scores:", cart_ent_cv_scores)
print("Mean Cross-Validation Score:", np.mean(cart_ent_cv_scores))


# In[114]:


modelEvaldf = pd.DataFrame(columns=['Model', 'ACCURACY'])
modelEvaldf.loc[len(modelEvaldf.index)] = ['Logistic Regression', lg_ac]
modelEvaldf.loc[len(modelEvaldf.index)] = ['SVC', sv_ac]
modelEvaldf.loc[len(modelEvaldf.index)] = ['Random Forest classifier', rf_ac]
modelEvaldf.loc[len(modelEvaldf.index)] = ['Random Forest classifier K-Fold', np.mean(rf_cv_scores)]
modelEvaldf.loc[len(modelEvaldf.index)] = ['CNN', round(cnn_ac[1], 3)]
modelEvaldf.loc[len(modelEvaldf.index)] = ['Decision Tree Gini', cartg_ac]
modelEvaldf.loc[len(modelEvaldf.index)] = ['Decision Tree Entropy K-Fold', np.mean(cart_ent_cv_scores)]
modelEvaldf.loc[len(modelEvaldf.index)] = ['Decision Tree Entropy', carte_ac]


# In[115]:


modelEvaldf


# In[ ]:




