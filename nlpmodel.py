import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("C:\\Users\\DELL\\Downloads\\nlpdeployment\\moviereviews.csv")
print('your df')
print(df.head())

df=df[df['review'].str.isspace()==False]

from sklearn.model_selection import train_test_split

X = df['review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

pipe = Pipeline([('tfidf', TfidfVectorizer()),('svc', LinearSVC()),])
pipe.fit(X_train, y_train) 

example_set=['This movie was a disaster']
print('prediction for example set')
print(pipe.predict(example_set))

pickle.dump(pipe,open('pipe.pkl','wb'))

#the following model predicts if a review is positive or negative 

