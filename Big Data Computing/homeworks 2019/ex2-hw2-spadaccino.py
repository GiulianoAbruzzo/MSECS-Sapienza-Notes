
import sys
from time import time

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# Download additional components for word tokenization
nltk.download("punkt")
nltk.download('wordnet')


# If USE_PCA=False then uses plain SVD
USE_PCA = False


# Categories of the dataset. Use `categories=None` to load all of them
categories = [
    "comp.graphics",
    "rec.motorcycles",
    "sci.med",
    "misc.forsale",
    "talk.politics.guns",
    "talk.religion.misc"
]
#categories = None    #decomment this to use all categories
print(f"Loading 20 newsgroups dataset for categories: {categories if categories else 'all'}")

# Load dataset
dataset = fetch_20newsgroups(subset='all', categories=categories, 
    shuffle=False, remove=('headers', 'footers', 'quotes'))
labels = dataset.target
true_k = len(np.unique(labels))
print(f"true_k: {true_k}")


# Process words
lemmatizer = WordNetLemmatizer()
for i in range(len(dataset.data)):
    word_list = word_tokenize(dataset.data[i])
    lemmatized_doc = ""
    for word in word_list:
        lemmatized_doc = lemmatized_doc + " " + lemmatizer.lemmatize(word)
    dataset.data[i] = lemmatized_doc

vectorizer = TfidfVectorizer(stop_words='english') ## Corpus is in English
X = vectorizer.fit_transform(dataset.data).astype(np.float32)
print(f"Data loaded and word processed, shape: {X.shape}")




# k-means on raw data
km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))

def print_metrics(true_labels, labels):
    print("METRICS")
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, true_labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, true_labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, true_labels))
    print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, true_labels))
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, true_labels, sample_size=1000))
    print("-------")
print_metrics(labels, km.labels_)

# Print major components for each centroid
print("APPROACH 1. K-MEANS ON RAW DATA, terms:")
centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i, end='')
    for ind in centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()
print("-------------------------------------")



# SVD/PCA
print("The original data have", X.shape[1], "dimensions/features/terms")
r = true_k
t0 = time()
if not USE_PCA: # Plain SVD
  svd = TruncatedSVD(r)
  decomposition = svd
else:  # PCA
  X = X.todense()
  pca = PCA(r, copy=False)
  decomposition = pca
normalizer = Normalizer(copy=False)
lsa = make_pipeline(decomposition, normalizer)
Y = lsa.fit_transform(X)
print("done in %fs" % (time() - t0))

print("The number of documents is still", Y.shape[0])
print("The number of dimension has become", Y.shape[1])

# Top terms corresponding to each principal direction
print(f"APPROACH 2. {'PCA' if USE_PCA else 'SVD'} DECOMPOSITION, top terms of principal directions:")
terms = vectorizer.get_feature_names()
for i, comp in enumerate(decomposition.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    s = ""
    for t in sorted_terms:
        s += t[0] + " "
    print(s)
print("-----------------------")

# Worst terms corresponding to principal directions
print("TOP UNRELATED TERMS TO PRINCIPAL DIRECTIONS")
terms = vectorizer.get_feature_names()
for i in range(decomposition.components_.shape[0]):
    terms_comp = [[terms[j], decomposition.components_[i][j]] for j in range(decomposition.components_.shape[1])]
    asc_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:10]
    des_terms = sorted(terms_comp, key= lambda x:x[1], reverse=False)[:10]
    print("Topic "+str(i)+": ")
    s = ""
    for t in asc_terms:
        s += t[0] + " "
    print(s)
    s = ""
    for t in des_terms:
        s += t[0] + " "
    print(s)
print("-------------------------------------------")

# k-means on principal directions
km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100)
t0 = time()
km.fit(Y)
print_metrics(labels, km.labels_)
print(km.cluster_centers_.shape)
print(f"done in {time() - t0:.3f}s")

# Retrieve original centroids
original_centroids = decomposition.inverse_transform(km.cluster_centers_)
print(original_centroids.shape) ## Just a sanity check
for i in range(original_centroids.shape[0]):
    original_centroids[i] = np.array([x for x in original_centroids[i]])
decomposition_centroids = original_centroids.argsort()[:, ::-1]

# Top components for each centroid
print("APPROACH 3. K-MEANS ON PRINCIPAL COMPONENTS, terms:")
for i in range(true_k):
    print("Cluster %d:" % i, end='')
    for ind in decomposition_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()
print("-------------------------------------------------")
