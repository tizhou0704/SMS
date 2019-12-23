import os
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk import PorterStemmer as Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#find dataset location
# os.listdir("../final_project")

# data input
df = pd.read_csv('../final_project/spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df.head().groupby('label').describe()
# df.groupby('label').describe()
print('total number of Ham and Spam message: ')
print(df.groupby('label').size())



#data cleaning function
def process(text):
    # lowercase it
    text = text.lower()
    # remove punctuation
    text = ''.join([t for t in text if t not in string.punctuation])
    # remove stopwords
    text = [t for t in text.split() if t not in stopwords.words('english')]
    # stemming
    st = Stemmer()
    text = [st.stem(t) for t in text]
    # return token list
    return text


# test cleaning function with single message
process('It\'s holiday and we are playing cricket. Jeff is playing very well!!!')
# Test with our dataset
df['message'][:20].apply(process)





tfidfv = TfidfVectorizer(analyzer=process)
data = tfidfv.fit_transform(df['message'])
mess = df.iloc[2]['message']
print(5)
print(mess)
print(tfidfv.transform([mess]))
print(6)
j = tfidfv.transform([mess]).toarray()[0]


print('index\tidf\ttfidf\tterm')
for i in range(len(j)):
    if j[i] != 0:
        print(i, format(tfidfv.idf_[i], '.4f'), format(j[i], '.4f'), tfidfv.get_feature_names()[i],sep='\t')



spam_filter = Pipeline([
    ('vectorizer', TfidfVectorizer(analyzer=process)), # messages to weighted TFIDF score
    ('classifier', MultinomialNB())                    # train on TFIDF vectors with Naive Bayes
])




x_train, x_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.30, random_state = 21)
spam_filter.fit(x_train, y_train)
predictions = spam_filter.predict(x_test)
count = 0
for i in range(len(y_test)):
    if y_test.iloc[i] != predictions[i]:
        count += 1

print(7)
print('Total number of test cases', len(y_test))
print('Number of wrong of predictions', count)
print(8)
x_test[y_test != predictions]




print(classification_report(predictions, y_test))







def detect_spam(s):
    return spam_filter.predict([s])[0]
detect_spam('Your cash-balance is currently 500 pounds - to maximize your cash-in now, send COLLECT to 83600.')
