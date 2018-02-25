#programmer: Mohammad J. Nourbakhsh
import computeConfSim as ccs
import sys, os


def print_knn_for_single_conf(csdct, c1, conf,  resultsDir, N=10):
	st = ""
	ii = 1
	for c2 in sorted(csdct[c1], key=csdct[c1].get, reverse=True): # c2 is a conf_id
		st+= "\t" + str(ii) + ". " + conf[c2] +  " ID: "  + str(c2) +  " CosSim: " +  str(csdct[c1][c2]) +"\n"
		ii +=1
		if ii > N:
			break
	return st

def print_knn_results( conf, cstfidf, cslsi,resultsDir,  N=10 ):

	with open(resultsDir +"/knn-results.txt", "wb") as f:
		for c1 in sorted(conf.keys()): # c1 is a conf_id
			f.write(  "-------\nConf: "+ conf[c1] + " ID:" + str(c1) +"\n")
			f.write( str(N) + "-nearest neighbours (most similar)  using  TFIDF vectors and cosine similarity: \n")
			st = print_knn_for_single_conf ( cstfidf, c1, conf, resultsDir, N)
			f.write(st)
			f.write( str(N) + "-nearest  neighbours (most similar) using LSI vectors and cosine similarity:\n")
			st = print_knn_for_single_conf( cslsi, c1, conf, resultsDir, N)
			f.write( st +  "\n\n")



def main():
	cwd = os.getcwd()
	datasetPath = cwd + "/four_area_dataset"
	outputDir = cwd +"/gensimOutputDir"
	resultsDir = cwd +"/results"
	print ("\n----------- DESCRIPTION")
	print ("This code computes the  K-Nearest Neighbour (KNN) algorithm for academic conferences like WWW, KDD, CVPR, etc. It works by:" +\
	 "\n1- For each conference in the dataset, it finds the papers published in that conference, and extracts their titles. \n The titles are concatenated and each conference is considered as a document containing those titles. " +\
	 "\n2- Using Python Gensim library, two types of vectors are created for each conference:" +\
	 "\n\t 2.a- Word-based representation using TFIDF   \n\t 2.b-Semantic representation using LSI vector " +\
	 "\n3- The K-nearest conferences for every conference in the dataset is found by:"+\
	 "\n\t 3.a- Computing cosine similarity between TFIDF vectors \n\t 3.b- Computing cosine similarity between LSI vectors  "+\
	 "\n\n-----------")
	conf, cstfidf, cslsi = ccs.start(datasetPath,outputDir)
	print ("\n\n-----------")
	N=input("The K-NN  algorithm needs the number of nearest neighbours.\nEnter an integer K  (less than or equal to 20) and press enter...\n")
	print ("Creating Results Folder")
	if not os.path.exists(resultsDir):
		os.makedirs(resultsDir)
		print ("Created: ", resultsDir)
	else:
		print>>sys.stderr, resultsDir," already exists."
	print ("Results for", N, "-neareset neighbours algorithm are stored in",  resultsDir)
	print_knn_results( conf, cstfidf, cslsi,  resultsDir, N = N)

if __name__ == "__main__":
	main()
