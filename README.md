# KNN-Algorithm-Python
//Programmer: Mohammad J. Nourbakhsh

This code computes the  K-Nearest Neighbour (KNN) algorithm for academic conferences like WWW, KDD, CVPR, etc. It works by:
1- For each conference in the dataset, it finds the papers published in that conference, and extracts their titles. 
 The titles are concatenated and each conference is considered as a document containing those titles. 
2- Using Python Gensim library, two types of vectors are created for each conference:
	 2.a- Word-based representation using TFIDF   
	 2.b-Semantic representation using LSI vector 
3- The K-nearest conferences for every conference in the dataset is found by:
	 3.a- Computing cosine similarity between TFIDF vectors 
	 3.b- Computing cosine similarity between LSI vectors  

 



The code uses gensim library for python. To install gensim run: 
easy_install -U gensim


To run the code: 
python knn.py
don’t forget to enter an integer K when the code asks. 
