from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords 
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.special import comb

import numpy as np
import pandas as pd
import os
import json
import wikipedia
import sklearn
import re
import scipy

ndocuments = 1000
clusterSize = 20
wikipedia.set_lang("en")  

#stemma e lemma i documenti
def documents_stem_lem(data):
    stemmed_totale = []
    lemmatized_totale = []  
    
    for i in range(ndocuments):
        print("************************ PARSING DOCUMENT " + str(i) + " ************************")
        
        #documento originale tokenizato
        document_tokenized = word_tokenize(data[i])
        
        #modelli di stemming e lemmatizazione
        lemmatizer = WordNetLemmatizer()
        porter = PorterStemmer()
        stop_words = set(stopwords.words('english')) 

        #creiamo documenti di stemming e lemmatizazione
        punctuations="?:!.,;{}'%()[]<>*-\/&^$#@|"
        stemmed_words=[]
        lemmatized_words=[]
        
        for word in document_tokenized:
            word = word.lower()
            
            #eliminiamo le parole con caratteri speciali
            if not word in punctuations and word not in stop_words:
                if (bool(re.match('^[a-zA-Z0-9]*$',word))==True):
                    lem = lemmatizer.lemmatize(word, pos='v')
                    stem = porter.stem(lem)
                
                    stemmed_words.append(stem)
                    lemmatized_words.append(lem)
        
        #lista di liste di tutti i documenti
        stemmed_totale.append(stemmed_words)
        lemmatized_totale.append(lemmatized_words)
            
    #salviamo i documenti nei file
    f = open("documents_stemmed.json", "w+")
    json_data = json.dumps(stemmed_totale)
    f.write(json_data)
    f.close()

    f = open("documents_lemmatized.json", "w+")
    json_data = json.dumps(lemmatized_totale)
    f.write(json_data)
    f.close()
    
    return stemmed_totale, lemmatized_totale

#cerca i termini e le categorie su wikipedia
def wikiped(lemmatized_documents):
    dizionario_term_titolo = {}
    dizionario_term_categorie = {}
    lista_errori = []
    lista_disambiguation = []
    i=0;
        
    #carico stemmed/lemmatized/dizionario documents
    dizionario_term_titolo_path = Path(os.getcwd()+"/dizionario_term_titolo.json")    
    dizionario_term_categorie_path = Path(os.getcwd()+"/dizionario_term_categorie.json")  
    wiki_errori_path = Path(os.getcwd()+"/wiki-errori.json")
    wiki_dis_path = Path(os.getcwd()+"/wiki-disambiguations.json")
    
    if dizionario_term_titolo_path.exists() and dizionario_term_categorie_path.exists() and wiki_errori_path.exists() and wiki_dis_path.exists():
        print("************************ JSON OLD TROVATI ************************")
        with open("dizionario_term_titolo.json", 'r+') as myfile:
            data=myfile.read()
        dizionario_term_titolo = json.loads(data)    
        
        with open("dizionario_term_categorie.json", 'r+') as myfile:
            data=myfile.read()
        dizionario_term_categorie = json.loads(data) 
        
        with open("wiki-errori.json", 'r+') as myfile:
            data=myfile.read()
        lista_errori = json.loads(data) 
        
        with open("wiki-disambiguations.json", 'r+') as myfile:
            data=myfile.read()
        lista_disambiguation = json.loads(data) 

    #scorriamo tutti i documenti lemmatizzati
    for document in lemmatized_documents:
        print("\n************************ START WIKIPEDIA DOCUMENT " + str(i) + " ************************")
    
        #per ogni termine nel documento lo cerchiamo su wikipedia
        for term in document:
        
            #usiamo il try perchè wikipedia genera errori se non viene trovato un termine
            try:  
                #se il termine non è già presente nel nostro dizionario lo andiamo a cercare
                if term not in dizionario_term_titolo and term not in lista_errori and term not in lista_disambiguation:
                    print("///// WIKIPEDIA SEARCHING " + term + " /////")
                
                    #prendi le informazioni su wikipedia di quel termine
                    page = wikipedia.WikipediaPage(str(term))
                    
                    #salva il titolo trovato su quel termine
                    dizionario_term_titolo[term] = str(page.title)
                    
                    #rimuovi categorie amministrative di wikipedia che non ci servono
                    filters = ['category' ,"needing confirmation" , 'use british english' ,'webarchive' ,"wiktionary" ,'needing' ,'ambiguous' ,"incomplete" , 'pages' ,'error' ,"dmy" ,'use american english' ,"stubs" ,'use'  ,'articles' ,'reference' ,'cs1' , 'links' ,'disputes' ,'mdy' ,"lists" ,'confirmation' , 'engvarb' ,'wikidata' ,'wikipedia' ,'cs1:' ,"redirects" , 'pages using timeline' ,'british english']
                    
                    #salva le categorie a cui appartiene quel termine su wikipedia
                    categories = []
                    
                    for c in page.categories:
                        #uso una flag per controllare, appena una delle parole della filter list è nella categoria la scarto
                        flag_filter = True
                        for f in filters:
                            if f in c.lower():
                                flag_filter = False
                                break
                        if flag_filter == True: 
                            categories.append(c.lower())
                            
                    #lo aggiungiamo al dizionario titoli-categorie
                    dizionario_term_categorie[str(page.title)] = categories
                
            #se ci sono errori salvameli su queste due liste    
            except wikipedia.exceptions.PageError:
                print("///// ERROR WHILE SEARCHING " + term + " /////")
                lista_errori.append(term)
                continue

            except wikipedia.exceptions.DisambiguationError:
                print("///// DISAMBIGUATION WHILE SEARCHING " + term + " /////")
                lista_disambiguation.append(term)
                continue    
        i=i+1;
        
        f = open("last_document_scrapped.txt", "w+")
        f.write(str(i))
        f.close()
        
        f = open("dizionario_term_titolo.json", "w+")
        json_data = json.dumps(dizionario_term_titolo)
        f.write(json_data)
        f.close()

        f = open("dizionario_term_categorie.json", "w+")
        json_data = json.dumps(dizionario_term_categorie)
        f.write(json_data)
        f.close()

        f = open("wiki-errori.json", "w+")
        json_data = json.dumps(lista_errori)
        f.write(json_data)
        f.close()   
        
        f = open("wiki-disambiguations.json", "w+")
        json_data = json.dumps(lista_disambiguation)
        f.write(json_data)
        f.close() 
    return dizionario_term_titolo, dizionario_term_categorie, lista_errori, lista_disambiguation 

#carica i json se già esistono
def load_jsons(data):
    #carico stemmed/lemmatized/dizionario documents
    document_stemmed_path = Path(os.getcwd()+"/documents_stemmed.json")    
    document_lemmatized_path = Path(os.getcwd()+"/documents_lemmatized.json")  
    
    if document_stemmed_path.exists() and document_lemmatized_path.exists():    
    
        print("************************ JSON DEI DOCUMENTI TROVATI ************************")
        with open("documents_stemmed.json", 'r+') as myfile:
            data=myfile.read()
        stemmed_documents = json.loads(data)    
        
        with open("documents_lemmatized.json", 'r+') as myfile:
            data=myfile.read()
        lemmatized_documents = json.loads(data) 
        
    #se non esistono li creo
    else:
        print("************************ JSON DEI DOCUMENTI NON TROVATI ************************")
        stemmed_documents, lemmatized_documents = documents_stem_lem(data)
    
    #carico stemmed/lemmatized/dizionario documents
    dizionario_term_titolo_path = Path(os.getcwd()+"/dizionario_term_titolo.json")    
    dizionario_term_categorie_path = Path(os.getcwd()+"/dizionario_term_categorie.json")  
    wiki_errori_path = Path(os.getcwd()+"/wiki-errori.json")
    wiki_dis_path = Path(os.getcwd()+"/wiki-disambiguations.json")
    
    last_scrapped = Path(os.getcwd()+"/last_document_scrapped.txt")
    if last_scrapped.exists():
        with open("last_document_scrapped.txt", 'r+') as myfile:
            data=myfile.read()  
            #print(data, ndocuments)
    else:
        data=0
        #print("niente zi")
    
    if dizionario_term_titolo_path.exists() and dizionario_term_categorie_path.exists() and wiki_errori_path.exists() and wiki_dis_path.exists() and float(data)==ndocuments:
    
        print("************************ JSON DI WIKIPEDIA TROVATI ************************")
        with open("dizionario_term_titolo.json", 'r+') as myfile:
            data=myfile.read()
        dizionario_term_titolo = json.loads(data)    
        
        with open("dizionario_term_categorie.json", 'r+') as myfile:
            data=myfile.read()
        dizionario_term_categorie = json.loads(data) 
        
        with open("wiki-errori.json", 'r+') as myfile:
            data=myfile.read()
        lista_errori = json.loads(data) 
        
        with open("wiki-disambiguations.json", 'r+') as myfile:
            data=myfile.read()
        lista_disambiguation = json.loads(data) 
         
    #se non esistono li creo
    else:
        print("************************ JSON DI WIKIPEDIA NON TROVATI ************************")
        dizionario_term_titolo, dizionario_term_categorie, lista_errori, lista_disambiguation = wikiped(lemmatized_documents)
        
    return stemmed_documents, lemmatized_documents, dizionario_term_titolo, dizionario_term_categorie, lista_errori, lista_disambiguation

#funzione di appoggio per il vectorizer
def useless(document):
    return document
    
#calcola la distanza cos tra i documenti
def cos_distance(documents):
    vectorizer = TfidfVectorizer()
    tfidf_vectors = tfidf.fit_transform(documents)
    return sklearn.metrics.pairwise.cosine_distances(tfidf_vectors, Y=None)     
 
#crea lista di documenti di solo concetti 
def sistemaConcettiWikipedia(lemmatized_documents, dizionario_term_titolo):
    #creiamo una lista dei documenti di soli concetti
    concept_documents = []
    
    #cerchiamo in ogni documento lemmatizato, e sostituiamo le parole con i concetti di wikipedia
    for document in lemmatized_documents:
        new_doc = []
        for word in document:
            if word in dizionario_term_titolo:
                new_doc.append(dizionario_term_titolo[word])
                
        #lista dei nuovi documenti con concetti
        concept_documents.append(new_doc)
    return concept_documents 
 
#crea lista di documenti di solo categorie
def sistemaCategorieWikipedia(documents, dizionario_term_categorie):
    #creiamo una lista dei documenti di soli categorie
    category_documents = []

    #cerchiamo in ogni documento e sostituiamo le parole con le categorie
    for document in documents:
        new_doc = []
        for word in document:
            #salviamo le categorie associate a quella parola
            categories = dizionario_term_categorie[word]
            
            #se ho zero categorie per quella parola la salto
            if len(categories) == 0:
                continue
                
            #se ho piu categorie allora le scorro 
            else:
                for i in range(len(categories)):
                    new_doc.append(categories[i])
        category_documents.append(new_doc)
    return category_documents 
 
#crea dizionario cluster-label articoli 
def crea_dizionario_cluster_categorie(clusters):
    #crea dizionario cluster: documenti che appartengono a quel cluster
    lista_cluster = clusters.tolist()
    
    dizionario_cluster_documenti={}
    for i in range(len(lista_cluster)):
        if lista_cluster[i] in dizionario_cluster_documenti:
            dizionario_cluster_documenti[lista_cluster[i]].append(i)
            
        if lista_cluster[i] not in dizionario_cluster_documenti:
            dizionario_cluster_documenti[lista_cluster[i]] = []
            dizionario_cluster_documenti[lista_cluster[i]].append(i)     
     
    #crea dizionario cluster: categoria articolo dei documenti che appartengono a quel cluster
    dizionario_cluster_categorie = {}
    for i in range(len(dizionario_cluster_documenti)):
        for element in dizionario_cluster_documenti[i]:
            if i in dizionario_cluster_categorie:
                dizionario_cluster_categorie[i].append(topics[element])
                
            if i not in dizionario_cluster_categorie:
                dizionario_cluster_categorie[i] = []
                dizionario_cluster_categorie[i].append(topics[element])
    return dizionario_cluster_categorie        

#formula della similarità definita nel paper, gamma introdotta per concept-category dove la matrice di word non deve contare
def similarita(wordDistanceMatrix, conceptDistanceMatrix, categoryDistanceMatrix, alpha, beta, gamma):
    matrice_similarita = wordDistanceMatrix*gamma +conceptDistanceMatrix*alpha + categoryDistanceMatrix*beta
    return matrice_similarita
      
#ogni cluster è assegnato alla categoria piu frequente del cluster, e l'accuratezza di questa assegnazione è
#misurata contando il numero di documenti assegnati corretti divisi dal numero totale
def purity(clusters):
    diz = crea_dizionario_cluster_categorie(clusters)
    
    #lista delle categorie max per ogni cluster
    massimo_cluster = []
    
    #lista del numero di occorenze di categorie
    occorrenze_massimo = []
    
    #inseriamo nella lista il numero di occorrenze di elementi alla categoria massima presente in quel cluster
    for i in range(clusterSize):
        cluster = diz[i]
        massimo_cluster.append(max(set(cluster), key = cluster.count))
        occorrenze_massimo.append(cluster.count(massimo_cluster[i]))
        
    #totale diviso il numero di documenti
    tot=0
    for i in range(clusterSize):
        tot += occorrenze_massimo[i]
    
    purity = tot/ndocuments
    return purity

#formula del fscore per il clustering presa da: https://stats.stackexchange.com/a/157385
def FScore(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum() for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    
    precision = tp/ (tp+fp)
    recall = tp/(tp+fn)

    fscore = (2*precision*recall)/(precision+recall)
    return fscore

#cerco i valori piu alti per word-concept-category    
def scoreMax(fscore, nmi, purity):
    #ritrova l'id con i valori massimi
    indici = []
    indici.append(fscore.index(max(fscore)))
    indici.append(nmi.index(max(nmi)))
    indici.append(purity.index(max(purity)))
    result = max(set(indici), key = indici.count) 
    return result    
 
#calcola i totali degli score e le best alpha e beta 
def calcola_totali(wordDistanceMatrix,conceptDistanceMatrix,categoryDistanceMatrix):
    #WORD-CONCEPT cicliamo dieci volte come dice il paper e poi andremo a fare una media, in questa lista mettiamo i vari clusters dei dieci cicli
    lista_word_concept = []
    alpha = 0.1
    beta = 0

    for i in range (10):
        similarityMatrix = similarita(wordDistanceMatrix,conceptDistanceMatrix,categoryDistanceMatrix,alpha,beta,1)
        lista_word_concept.append(AgglomerativeClustering(affinity='precomputed',n_clusters=clusterSize,linkage='complete').fit(similarityMatrix))
        alpha = alpha+0.1

    #WORD-CATEGORY cicliamo dieci volte come dice il paper e poi andremo a fare una media
    lista_word_category=[]
    alpha =0
    beta =0.1
    for i in range (10):
        similarityMatrix = similarita(wordDistanceMatrix,conceptDistanceMatrix,categoryDistanceMatrix,alpha,beta,1)
        lista_word_category.append(AgglomerativeClustering(affinity='precomputed', n_clusters=clusterSize, linkage='complete').fit(similarityMatrix))
        beta = beta +0.1

    #sommiamo i vari score per WORD-CONCEPT
    tot_fscore_word_concept = 0
    tot_purity_word_concept = 0
    tot_nmi_word_concept = 0

    #sommiamo i vari score per WORD-CATEGORY
    tot_fscore_word_category = 0
    tot_purity_word_category = 0
    tot_nmi_word_category = 0

    #usiamo queste liste per tenere traccia dei vari score e trovare il migliore da usare dopo per WORD-CONCEPT-CATEGORY
    lista_fscore_word_concept = []
    lista_purity_word_concept = []
    lista_nmi_word_concept = []

    lista_fscore_word_category = []
    lista_purity_word_category = []
    lista_nmi_word_category = []

    for i in range(10):
        tot_fscore_word_concept+=FScore(lista_word_concept[i].labels_,topics)
        tot_purity_word_concept+=purity(lista_word_concept[i].labels_)
        tot_nmi_word_concept+=normalized_mutual_info_score(topics,lista_word_concept[i].labels_)

        tot_fscore_word_category+=FScore(lista_word_category[i].labels_,topics)
        tot_purity_word_category+=purity(lista_word_category[i].labels_)
        tot_nmi_word_category+=normalized_mutual_info_score(topics,lista_word_category[i].labels_)

        lista_fscore_word_concept.append(FScore(lista_word_concept[i].labels_,topics))
        lista_purity_word_concept.append(purity(lista_word_concept[i].labels_))
        lista_nmi_word_concept.append(normalized_mutual_info_score(topics,lista_word_concept[i].labels_))

        lista_fscore_word_category.append(FScore(lista_word_concept[i].labels_,topics))
        lista_purity_word_category.append(purity(lista_word_concept[i].labels_))
        lista_nmi_word_category.append(normalized_mutual_info_score(topics,lista_word_concept[i].labels_))

    #troviamo i migliori alpha e beta
    bestScore = scoreMax(lista_fscore_word_concept, lista_nmi_word_concept, lista_purity_word_concept)
    bestAlpha = bestScore*0.1+0.1

    bestScore = scoreMax(lista_fscore_word_category, lista_nmi_word_category, lista_purity_word_category)
    bestBeta = bestScore*0.1+0.1

    #CONCEPT-CATEGORY dobbiamo ciclare 10 volte per entrambi i tipi con 0 come gamma
    lista_concept_category =[]
    alpha=0.1
    beta=0.1
    for x in range(10):
        for y in range(10):
            similarityMatrix = similarita(wordDistanceMatrix, conceptDistanceMatrix, categoryDistanceMatrix, alpha, beta,0)
            lista_concept_category.append(AgglomerativeClustering(affinity='precomputed', n_clusters=clusterSize, linkage='complete').fit(similarityMatrix))
            beta=beta+0.1
        alpha = alpha+0.1

    #sommo i vari score per poi fare la media
    tot_fscore_concept_category=0
    tot_purity_concept_category=0
    tot_nmi_concept_category=0

    for i in range (100):
        tot_fscore_concept_category+=FScore(lista_concept_category[i].labels_,topics)
        tot_purity_concept_category+=purity(lista_concept_category[i].labels_)
        tot_nmi_concept_category+=normalized_mutual_info_score(topics,lista_concept_category[i].labels_)

    return tot_purity_word_concept, tot_nmi_word_concept, tot_fscore_word_concept, tot_purity_word_category, tot_nmi_word_category, tot_fscore_word_category, tot_purity_concept_category, tot_nmi_concept_category, tot_fscore_concept_category, bestAlpha, bestBeta
    
#inizializza dataset e vettorizatore 
newsgroup_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
tfidf = TfidfVectorizer(analyzer='word',tokenizer=useless,preprocessor=useless,token_pattern=None,use_idf=True,sublinear_tf=False) 

#contiene tutti gli articoli
data = []

#contiene la categoria associata a quel articolo
topics=[]

#creiamo dataset e categorie associate
for i in range(ndocuments):
    data.append(newsgroup_train.data[i])
    topics.append(newsgroup_train.target[i])   
    
#se non esistono questi json li creiamo
stemmed_documents, lemmatized_documents, dizionario_term_titolo, dizionario_term_categorie, lista_errori, lista_disambiguation = load_jsons(data)
print("************************ ALL DOCUMENTS PARSED ************************")
print("************************ ALL TERMS SEARCHED ON WIKIPEDIA ************************")

#per il caso base usiamo direttamente i documenti stemmati originali (stemmed_documents) per gli altri devo sistemarli prima di creare la matrice
#utilizzo solo i documenti necessari, escludo se ne ho di più
stemmed_documents = stemmed_documents[:ndocuments]
lemmatized_documents = lemmatized_documents[:ndocuments]

#nuovi documenti di soli concetti o sole categorie
conceptDocuments = sistemaConcettiWikipedia(lemmatized_documents, dizionario_term_titolo)
categoryDocuments = sistemaCategorieWikipedia(conceptDocuments, dizionario_term_categorie)

#calcoliamo la matrice delle distanze tra i vari documenti
wordDistanceMatrix = cos_distance(stemmed_documents)
conceptDistanceMatrix = cos_distance(conceptDocuments)
categoryDistanceMatrix = cos_distance(categoryDocuments)

#facciamo dei clusters completi con la matrice delle distanze
word_agglomerato = AgglomerativeClustering(affinity='precomputed', n_clusters=clusterSize, linkage='complete').fit(wordDistanceMatrix)
concept_agglomerato = AgglomerativeClustering(affinity='precomputed', n_clusters=clusterSize, linkage='complete').fit(conceptDistanceMatrix)
category_agglomerato = AgglomerativeClustering(affinity='precomputed', n_clusters=clusterSize, linkage='complete').fit(categoryDistanceMatrix)

#calcola totali degli score (dopo farò le medie) e best alpha e best beta
tot_purity_word_concept, tot_nmi_word_concept, tot_fscore_word_concept, tot_purity_word_category, tot_nmi_word_category, tot_fscore_word_category, tot_purity_concept_category, tot_nmi_concept_category, tot_fscore_concept_category, bestAlpha, bestBeta = calcola_totali(wordDistanceMatrix,conceptDistanceMatrix,categoryDistanceMatrix)

#WORD-CONCEPT-CATEGORY utilizzando la bestAlpha e la bestBeta trovati
similarityMatrix = similarita(wordDistanceMatrix, conceptDistanceMatrix, categoryDistanceMatrix, bestAlpha, bestBeta,1)
word_concept_category=AgglomerativeClustering(affinity='precomputed', n_clusters=20, linkage='complete').fit(similarityMatrix) 

print("\n************************ RISULTATI ************************")
print("\n************************ "+str(ndocuments)+" ************************")
print("WORD")
print("purity: "+ str(purity(word_agglomerato.labels_)))
print("nmi: "+ str(normalized_mutual_info_score(topics,word_agglomerato.labels_)))
print("fscore: "+ str(FScore(word_agglomerato.labels_,topics)))
print("\n")

print("CONCEPT")
print("purity: "+ str(purity(concept_agglomerato.labels_)))
print("nmi: "+ str(normalized_mutual_info_score(topics,concept_agglomerato.labels_)))
print("fscore: "+ str(FScore(concept_agglomerato.labels_,topics)))
print("\n")

print("CATEGORY")
print("purity: "+ str(purity(category_agglomerato.labels_)))
print("nmi: "+ str(normalized_mutual_info_score(topics,category_agglomerato.labels_)))         
print("fscore: "+ str(FScore(category_agglomerato.labels_,topics)))
print("\n")

print("WORD-CONCEPT")
print("purity: "+ str(tot_purity_word_concept/10))
print("nmi: "+ str(tot_nmi_word_concept/10))
print("fscore: "+ str(tot_fscore_word_concept/10))
print("\n")

print("WORD-CATEGORY")
print("purity: "+ str(tot_purity_word_category/10))
print("nmi: "+ str(tot_nmi_word_category/10))
print("fscore: "+ str(tot_fscore_word_category/10))
print("\n")

print("CONCEPT-CATEGORY")
print("purity: "+ str(tot_purity_concept_category/100))
print("nmi: "+ str(tot_nmi_concept_category/100))
print("fscore: "+ str(tot_fscore_concept_category/100))
print("\n")

print("WORD-CONCEPT-CATEGORY")
print("purity: "+ str(purity(word_concept_category.labels_)))
print("nmi: "+ str(normalized_mutual_info_score(topics,word_concept_category.labels_)))
print("fscore: "+ str(FScore(word_concept_category.labels_,topics)))