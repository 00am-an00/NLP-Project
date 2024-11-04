import pandas as pd  # data preprocessing
import nltk
nltk.download('wordnet')
import itertools  # confusion matrix
import string
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt
import requests
import zipfile
from io import BytesIO
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re


# Download and extract the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip"
response = requests.get(url)

if response.status_code == 200:
    with zipfile.ZipFile(BytesIO(response.content)) as thezip:
        with thezip.open('drugsComTrain_raw.tsv') as thefile:
            df = pd.read_csv(thefile, sep='\t')
else:
    print("Failed to download the file.")

# Display first few rows of data
df.head()
df.condition.value_counts()

# Filter dataset for specific conditions
df_train = df[(df['condition'] == 'Birth Control') | (df['condition'] == 'Depression') |
              (df['condition'] == 'High Blood Pressure') | (df['condition'] == 'Diabetes, Type 2')]
df.shape
df_train.shape

# Select features and target
X = df_train.drop(['Unnamed: 0', 'drugName', 'rating', 'date', 'usefulCount'], axis=1)
X.condition.value_counts()
X.head()

# Create subsets for word clouds
X_birth = X[X['condition'] == 'Birth Control']
X_dep = X[X['condition'] == 'Depression']
X_bp = X[X['condition'] == 'High Blood Pressure']
X_diab = X[X['condition'] == 'Diabetes, Type 2']

# Generate word clouds
plt.figure(figsize=(20, 20))
wc = WordCloud(max_words=500, width=1600, height=800).generate(" ".join(X_birth.review))
plt.imshow(wc, interpolation='bilinear')
plt.title('Word cloud for Birth control', fontsize=14)

plt.figure(figsize=(20, 20))
wc = WordCloud(max_words=500, width=1600, height=800).generate(" ".join(X_dep.review))
plt.imshow(wc, interpolation='bilinear')
plt.title('Word cloud for Depression', fontsize=14)

plt.figure(figsize=(20, 20))
wc = WordCloud(max_words=500, width=1600, height=800).generate(" ".join(X_bp.review))
plt.imshow(wc, interpolation='bilinear')
plt.title('Word cloud for High Blood Pressure', fontsize=14)

plt.figure(figsize=(20, 20))
wc = WordCloud(max_words=500, width=1600, height=800).generate(" ".join(X_diab.review))
plt.imshow(wc, interpolation='bilinear')
plt.title('Word cloud for Diabetes Type 2', fontsize=14)

# Text preprocessing
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if not w in stop]
    lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    return ' '.join(lemmitize_words)

X['review_clean'] = X['review'].apply(review_to_words)
X_feat = X['review_clean']
y = X['condition']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_feat, y, stratify=y, test_size=0.2, random_state=0)

# Confusion matrix plotting function
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Count Vectorizer and Naive Bayes
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)

mnb = MultinomialNB()
mnb.fit(count_train, y_train)
pred = mnb.predict(count_test)
score = metrics.accuracy_score(y_test, pred)
print("Naive Bayes accuracy: %0.3f" % score)

cm = metrics.confusion_matrix(y_test, pred, labels=['Birth Control', 'Depression', 'Diabetes, Type 2', 'High Blood Pressure'])
plot_confusion_matrix(cm, classes=['Birth Control', 'Depression', 'Diabetes, Type 2', 'High Blood Pressure'])

# Passive Aggressive Classifier with TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, ngram_range=(1,3))
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

pass_tf = PassiveAggressiveClassifier()
pass_tf.fit(tfidf_train, y_train)
pred = pass_tf.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
print("Passive Aggressive accuracy: %0.3f" % score)

cm = metrics.confusion_matrix(y_test, pred, labels=['Birth Control', 'Depression', 'Diabetes, Type 2', 'High Blood Pressure'])
plot_confusion_matrix(cm, classes=['Birth Control', 'Depression', 'Diabetes, Type 2', 'High Blood Pressure'])

# Display most informative features
def most_informative_feature_for_class(vectorizer, classifier, classlabel, n=10):
    labelid = list(classifier.classes_).index(classlabel)
    feature_names = vectorizer.get_feature_names_out()
    topn = sorted(zip(classifier.coef_[labelid], feature_names))[-n:]

    for coef, feat in topn:
        print(classlabel, feat, coef)

# Show informative features for each class
for condition in ['Birth Control', 'Depression', 'High Blood Pressure', 'Diabetes, Type 2']:
    most_informative_feature_for_class(tfidf_vectorizer, pass_tf, condition)

# Testing the model with custom reviews
test_reviews = [
    "I have only been on Tekturna for 9 days. The effect was immediate.",
    "This is the third med I tried for anxiety and mild depression.",
    "I just got diagnosed with type 2. My doctor prescribed Invokana."
]

for text in test_reviews:
    test = tfidf_vectorizer.transform([text])
    pred1 = pass_tf.predict(test)[0]
    print(f"Predicted condition for review '{text[:50]}...': {pred1}")
