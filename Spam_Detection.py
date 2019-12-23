import pandas as pd
import string
from nltk.corpus import stopwords
from nltk import PorterStemmer as Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# data input
df = pd.read_csv('../SMS/spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df.head().groupby('label').describe()
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
# check data format
print(mess)
print(tfidfv.transform([mess]))
j = tfidfv.transform([mess]).toarray()[0]

# print all results
print('index\tidf\ttfidf\tterm')
for i in range(len(j)):
    if j[i] != 0:
        print(i, format(tfidfv.idf_[i], '.4f'), format(j[i], '.4f'), tfidfv.get_feature_names()[i],sep='\t')



spam_filter_Naive_Bayes = Pipeline([
    ('vectorizer', TfidfVectorizer(analyzer=process)),                              # messages to weighted TFIDF score
    ('classifier', MultinomialNB())                                                 # train on TFIDF vectors with Naive Bayes
])


spam_filter_DecisionTree = Pipeline([
    ('vectorizer', TfidfVectorizer(analyzer=process)),                              # messages to weighted TFIDF score
    ('classifier', DecisionTreeClassifier())                                        # train on TFIDF vectors with Decision Tree
])

spam_filter_SVC = Pipeline([
    ('vectorizer', TfidfVectorizer(analyzer=process)),                              # messages to weighted TFIDF score
    ('classifier', SVC(kernel='linear'))                                            # train on TFIDF vectors with SVC
])



# set data fit models
x_train, x_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.30, random_state = 21)
spam_filter_Naive_Bayes.fit(x_train, y_train)
spam_filter_DecisionTree.fit(x_train, y_train)
spam_filter_SVC.fit(x_train, y_train)

predictions = spam_filter_Naive_Bayes.predict(x_test)
predictions_DT = spam_filter_DecisionTree.predict(x_test)
predictions_SVC = spam_filter_SVC.predict(x_test)
count = 0
count_DT = 0
count_SVC = 0
for i in range(len(y_test)):
    if y_test.iloc[i] != predictions[i]:
        count += 1

for i in range(len(y_test)):
    if y_test.iloc[i] != predictions_DT[i]:
        count_DT += 1


for i in range(len(y_test)):
    if y_test.iloc[i] != predictions_SVC[i]:
        count_SVC += 1

# print out result
print('result for model Naive Bayes:')
print('Total number of test cases', len(y_test))
print('Number of wrong of predictions', count)
x_test[y_test != predictions]
print(classification_report(predictions, y_test))


print('result for model decision Tree:')
print('Total number of test cases', len(y_test))
print('Number of wrong of predictions', count_DT)
x_test[y_test != predictions_DT]
print(classification_report(predictions_DT, y_test))


print('result for model SVC:')
print('Total number of test cases', len(y_test))
print('Number of wrong of predictions', count_SVC)
x_test[y_test != predictions_SVC]
print(classification_report(predictions_SVC, y_test))



# we choose the most efficient one--SVM model
def detect_spam(s):
    return spam_filter_SVC.predict([s])[0]
detect_spam('Your cash-balance is currently 500 pounds - to maximize your cash-in now, send COLLECT to 83600.')
