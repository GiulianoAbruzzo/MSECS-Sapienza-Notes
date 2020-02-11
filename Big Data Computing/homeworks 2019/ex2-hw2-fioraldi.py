from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans

import sys
from time import time
import random

import numpy as np

import nltk
nltk.download('punkt')
nltk.download('wordnet')

if len(sys.argv) < 3:
    print ("usage: ./ex2.py <SVD|PCA> <i>")
    exit(1)

alg = sys.argv[1].lower()
if alg == "svd":
    alg = TruncatedSVD
elif alg == "pca":
    alg = PCA
i = int(sys.argv[2])

#categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'misc.forsale', 'talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast', 'talk.religion.misc', 'alt.atheism', 'soc.religion.christian']

categories = []

topics = [
  ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x'],
  ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'],
  ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'],
  ['misc.forsale'],
  ['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast'],
  ['talk.religion.misc', 'alt.atheism', 'soc.religion.christian']
]

for j in range(len(topics)):
    random.shuffle(topics[j])

while len(categories) < i:
    for j in range(len(topics)):
        if len(topics[j]) > 0:
            categories.append(topics[j].pop())

print("Loading 20 newsgroups dataset for categories:")
print(categories)

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=False, remove=('headers', 'footers', 'quotes'))


labels = dataset.target
true_k = len(np.unique(labels)) ## This should be 3 in this example

lemmatizer = WordNetLemmatizer()
for i in range(len(dataset.data)):
    word_list = word_tokenize(dataset.data[i])
    lemmatized_doc = ""
    for word in word_list:
        lemmatized_doc = lemmatized_doc + " " + lemmatizer.lemmatize(word)
    dataset.data[i] = lemmatized_doc

vectorizer = TfidfVectorizer(stop_words='english') ## Corpus is in English
X = vectorizer.fit_transform(dataset.data)

print("The original data have", X.shape[1], "dimensions/features/terms")

print ("\n ********** KMEANS ONLY **********")

km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100)
t0 = time()
km.fit(X)
print("Done in %0.3fs" % (time() - t0))
print("KMeans centers shape:", km.cluster_centers_.shape)

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))

centroids = km.cluster_centers_.argsort()[:, ::-1] ## Indices of largest centroids' entries in descending order
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i, end='')
    for ind in centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()

print ("\n ********** DIM REDUCTION ONLY  **********")

r = true_k
t0 = time()
svd = alg(r)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
_X = X
if alg == PCA:
  _X = X.toarray()
Y = lsa.fit_transform(_X)
print("Done in %fs" % (time() - t0))

print("The number of documents is still", Y.shape[0])
print("The number of dimension has become", Y.shape[1])

terms = vectorizer.get_feature_names()

for i, comp in enumerate(svd.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:10]
    s = ""
    for t in sorted_terms:
        s += t[0] + " "
    print("Topic", i, ":", s)

print ("\n ********** DIM REDUCTION + KMEANS  **********")

km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100)
t0 = time()
km.fit(Y)
print("Done in %0.3fs" % (time() - t0))
print("KMeans centers shape:", km.cluster_centers_.shape)

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(Y, km.labels_, sample_size=1000))

original_centroids = svd.inverse_transform(km.cluster_centers_)
print(original_centroids.shape) ## Just a sanity check
for i in range(original_centroids.shape[0]):
    original_centroids[i] = np.array([x for x in original_centroids[i]])
svd_centroids = original_centroids.argsort()[:, ::-1]

for i in range(true_k):
    print("Cluster %d:" % i, end='')
    for ind in svd_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()




