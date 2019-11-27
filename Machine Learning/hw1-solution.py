import random
import os
import numpy
import imp
import pandas as pnd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

def main():
	#set of all application files
	all_files =  set(os.listdir('./feature_vectors'))
	
	#set of all malwares in the csv 	
	all_malware = set(pnd.read_csv('sha256_family.csv')['sha256'])

	#set of random malware and files	
	random_malware= set(random.sample(all_malware, int(len(all_malware)/5)))
	random_file= set(random.sample(all_files, int(len(all_files)/10)))
	non_malware= random_file.difference(random_malware)

	print ("Total of files: "+str(len(all_files)))
	print ("Total of malware: "+str(len(all_malware)))

	print ("Random file selected: "+str(len(random_file)))
	print ("Random malware selected: "+str(len(random_malware)))	

	#list of features
	features= {'permission','call','url','intent'}
	
	#dataset 
	dataset=list()

	#add all malware and non_malware in the dataset
	for malware in random_malware:
		file= open('./feature_vectors/'+malware)
		lines = file.readlines()
		key = ''
		for linea in lines:
			elemento = linea.strip().split('::')			
			if elemento[0] in features:
				key += elemento[1] + ','
		dataset.append([key,'malware'])
		file.close()

	for good in non_malware:
		file = open('./feature_vectors/'+good)
		lines= file.readlines()
		key=''
		for linea in lines:
			elemento= linea.strip().split('::')
			if elemento[0] in features:
				key += elemento[1] + ','
		dataset.append([key,'not_malware'])
		file.close()

	print("Created a dataset of dimension: "+ str(len(dataset)))
	class_list = ['malware', 'not_malware']
	
	#shuffle the dataset
	random.shuffle(dataset)
	
	#call kfold
	kfold_cross(dataset)
	
def kfold_cross(data):
	kf = KFold(n_splits=30)
	num_test=1
    
   	#for avg
	accuracy_tot=0
	precision_tot=0
	recall_tot=0
	f_p_tot=0
	f_measure_tot=0
	
	for train_index, test_index in kf.split(data):
		#create train and test list
		train_msg = list()
		train_classe = list()
		test_msg = list()
		test_classe = list()

		print("\nStarting test N°", str(num_test))
		for i in train_index:
			train_msg.append(data[i][0])
			train_classe.append(data[i][1])
		for j in test_index:
			test_msg.append(data[j][0])
			test_classe.append(data[j][1])

		#call count vectorizer that will create a matrix of token counts
		v= CountVectorizer()
		train_matrix = v.fit_transform(train_msg)
		
		#call multinomial naive bayes
		clf= MultinomialNB()
		clf.fit(train_matrix,train_classe)

		#create a matrix for the test list
		test_matrix = v.transform(test_msg)
		
		#predict with multinomial the test set
		predicted= clf.predict(test_matrix)

		#create a matrix of dim [2,2]
		conf_matrix = confusion_matrix(test_classe, predicted)
		tn, fp, fn, tp = conf_matrix.ravel()

		accuracy = (accuracy_score(test_classe, predicted)) * 100
		precision = precision_score(test_classe, predicted, pos_label='malware')
		recall = recall_score(test_classe, predicted, pos_label='malware')

		f_p_rate = fp/(fp+tn)
		f_measure = 2*(precision*recall)/(precision+recall)
        
		accuracy_tot+=accuracy
		precision_tot+=precision
		recall_tot+=recall
		f_p_tot+=f_p_rate
		f_measure_tot+= f_measure
        
		#print all the results
		print('Confusion matrix:')
		print(confusion_matrix(test_classe, predicted))
		print('Accuracy is: '+ str(accuracy)[:5]+'%')
		print('Precision: '+str(precision)[:5])
		print('Recall: '+str(recall)[:5])
		print('False-positive rating: '+str(f_p_rate)[:5])
		print('F-measure: '+str(f_measure)[:5])
		num_test+=1
    
	print('\nPerformance Evaluation, AVG Values:')
	print('Accuracy: '+ str(accuracy_tot/30)[:5]+'%')
	print('Precision: '+str(precision_tot/30)[:5])
	print('Recall: '+str(recall_tot/30)[:5])
	print('False-positive rating: '+str(f_p_tot/30)[:5])
	print('F-measure: '+str(f_measure_tot/30)[:5])
    
main()
