#programmer: Mohammad J. Nourbakhsh
import sys, os, getopt
import numpy as np

import gensim as gs
from gensim import corpora, models, similarities
from gensim.models import tfidfmodel
from gensim.models import lsimodel

def convertDoc(doc):
    docLemmas = gs.utils.lemmatize(doc);
    docLemmas = [gs.utils.any2unicode(word[:word.find('/')]) for word in docLemmas]
    return docLemmas

def computeVarianceMean(freqAuthorConf, a):
	myarray=np.array([freqAuthorConf[a][c] for c in freqAuthorConf[a]])
	return np.var(myarray), np.mean(myarray)



def parseDataset(path):

	print("Loading  and parsing dataset in  "+path+"...")

	path = path.strip()
	conf_file=open(path+"/"+"conf.txt")
	conf=dict()
	for line in conf_file.readlines():
		[conf_id,conf_name]= line.split()
		try:
			conf[ int(conf_id) ]= conf_name # chon conf_id reshtas
		except ValueError:
		    print ("not reading conf value in parseDataset")
		    pass

	i=0
	for conf_id in sorted(conf.keys()):
		    i+=1
		    #print i,conf_id,":",conf[ conf_id ]

	author_file = open(path+"/author.txt"); #encoding= 'utf8'
	author = dict()
	for line in author_file.readlines():
		    x = line.strip().split()
		    author_id = int(x[0])
		    author_name = x[1:]
		    author[ author_id ] = author_name

	i=0
	for author_id in sorted(author.keys()):
		    i+=1
		    #print i,author_id,":",author [author_id]

	term_file = open(path+"/term.txt");
	term = dict()
	for line in term_file.readlines():
		    x = line.strip().split()
		    term_id = int(x[0])
		    term_text= x[1:]
		    term[ term_id ] = term_text

	i=0
	for termi_d in sorted(term.keys()):
		    i+=1
		    #print i,term_id,":",term[ term_id ]

	year_file = open(path+"/year.txt");
	year= dict()
	for line in year_file.readlines():
		    x=line.strip().split()
		    str=' '.join(x[1:])
		    year[str]=int(x[0])

	i=0
	for title in sorted(year, key=year.get):
		    i+=1
		    #print i,title,year[title]


	conf_field_file = open(path+"/conf_field.txt");
	conf_field = dict()
	for line in conf_field_file.readlines():
		    x=line.strip().split()
		    conf_id = int(x[0])
		    field = x[2]
		    conf_field[ conf_id ]= field

	i=0
	for conf_id in sorted(conf_field.keys()):
		    i+=1
		    #print i,conf[conf_id],conf_field[conf_id]
			
			
	paper_file = open(path+"/paper.txt");
	paper=dict()
	for line in paper_file.readlines():
		    x = line.strip().split()
		    paper_id = int(x[0])
		    paper_title = x[1:] # x yek list e, x[0] = paper_id, x[1:] yani x[1] .. x[end], ke hamoon title paper mishe
		    paper[ paper_id ]= paper_title

	i=0
	for paper_id in sorted(paper.keys()):
		    i +=1
		    #print   i,paper_id,":",paper[paper_id]



	paper_conf_file = open(path+"/paper_conf.txt");
	paper_conf = dict()
	conf_paper = dict()
	for line in paper_conf_file.readlines():
		x=line.strip().split()
		paper_id = int(x[0]) # paper ID
		conf_id =int(x[1]) # conf ID
		paper_conf[ paper_id ]=conf_id
		
		if conf_id not in conf_paper.keys():
			conf_paper[conf_id] = dict()
			
		conf_paper[conf_id][paper_id] = paper[paper_id]
			
			
#		if conf_id not in conf_paper.keys():
#			conf_paper[conf_id] = []
#			
#		conf_paper[conf_id].append(paper_id)
			
#		    if conf_id  not in conf_paper.keys():
#		    	conf_paper[conf_id] = dict()
#
#		    conf_paper[conf_id][paper_id] =1
	new_file = open('C:/Users/mohammad/Dropbox/Neda Apply/emails/Mohammad/Mohammd replies/not sent/Reply to Keval Vora- sfu cs/my_codes/python/KNN-algorithm/gensimOutputDir'+'/aaaaaa.txt','w')
	print >>new_file, conf_paper
#	print(conf_paper)
	i=0
	for paper_id in sorted(paper_conf.keys()):
		    i+=1
		    #print i,paper[paper_id],conf[paper_conf[paper_id]]


#	paper_file = open(path+"/paper.txt");
#	paper=dict()
#	for line in paper_file.readlines():
#		    x = line.strip().split()
#		    paper_id = int(x[0])
#		    paper_title = x[1:] # x yek list e, x[0] = paper_id, x[1:] yani x[1] .. x[end], ke hamoon title paper mishe
#		    paper[ paper_id ]= paper_title
#
#	i=0
#	for paper_id in sorted(paper.keys()):
#		    i +=1
#		    #print   i,paper_id,":",paper[paper_id]

	return conf, author, term, year, conf_field, conf_paper, paper


def extractCorpusWithGensim(path, outputDir,conf, conf_paper, paper):

	print ("Using gensim library to create corpus. Creating gensim outputDir ")
	if not os.path.exists(outputDir):
		os.makedirs(outputDir)
		print ("\tCreated: "+ outputDir)
	else:
		print("\t" + outputDir+" already exists.")


	###############
#	nItems = len(conf)
	nItems = len(conf_paper)
	confId = nItems*[0]
	confIdtoIndex = {}
	print ("Number of conferences in dataset: "+ str(nItems))
	confLemmas = nItems*[['']]
	dictionary = corpora.Dictionary()

	conf_index=0
#	for conf_id in sorted(conf.keys()):
#		print ("\t adding conf_Name:"+conf[ conf_id ]+ " conf_ID:"+str( conf_id ))
#		confId[ conf_index] = conf_id
#		confIdtoIndex[ conf_id ] = conf_index
#		doc=""
##		for paper_id in conf_paper[ conf_id ].keys():
#            # inja darim vase conference c hameye paper_id hasho peyda mikonim va bezayer har paperid
#		for paper_id in conf_paper[ conf_id ].keys():	
#			paper_title = paper[paper_id]
#			#print "check this","c:",c, "p:",p,"paper", paper[p],
#
#			doc =doc+" "+' '.join(  paper_title )#till here
#			#print c, doc
#		confLemmas[ conf_index ] = convertDoc(doc)
#		#print confLemmas[ii]
#		conf_index +=1

	for conf_id in sorted(conf_paper.keys()):
		print ("\t adding conf_Name:"+conf[ conf_id ]+ " conf_ID:"+str( conf_id ))
		confId[ conf_index] = conf_id
		confIdtoIndex[ conf_id ] = conf_index
		doc=""
		for paper_id in conf_paper[conf_id].keys():
			doc =doc+" "+' '.join( conf_paper[conf_id][paper_id] )
			
		confLemmas[ conf_index ] = convertDoc(doc)
		conf_index +=1



	corpus =[]
#	dictionary.add_documents(confLemmas[confIdtoIndex[c]] for c in conf.keys());
	dictionary.add_documents(confLemmas[confIdtoIndex[c]] for c in conf_paper.keys());
	dictionary.save(outputDir+'/confDictionary.obj');
	dictionary.save_as_text(outputDir+'/confDictionary.txt');
	#creating bow model for each paper

#	for conf_id in conf:
	for conf_id in conf_paper:
		conf_index = confIdtoIndex[conf_id]
		doc = confLemmas[conf_index]
		try:
			vector =dictionary.doc2bow(doc)
		except UnicodeDecodeError:
			print >> sys.stderr, "Error: ", conf_id, docLemmas[conf_index]

		corpus.append( vector )
	corpora.MmCorpus.serialize(outputDir+'/corpus.mm', corpus)

	print confId
	return confId, confIdtoIndex, corpus, dictionary


def getTfidfLsiSims(corpus, confId, confIdtoIndex, dictionary, outputDir):
	print ("Using gensim to get TFIDF vector and LSI vector for conferences in corpus ")
	#tfidf
	tfidf = tfidfmodel.TfidfModel(corpus) # initialize a tfidf transformation for corpus
	corpus_tfidf = tfidf[corpus] # get tfidf vectors
	#lsi
	lsi = lsimodel.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=4) # initialize an LSI transformation for corpus, with number of topics = 4
	corpus_lsi = lsi[corpus_tfidf]


	####### not important, just printing
	print ("Printing TF-IDF vectors in "+ outputDir+'/conffTFIDF.txt')
	fTFIDFFile = open(outputDir+'/conffTFIDF.txt','w')
	j=0
	for doc in corpus_tfidf:
		print >>fTFIDFFile, confId[j] ,doc
		j = j+1
		if j % 100 == 0:
			print(j)
	tfidf.save(outputDir+'/conftfidf.mod')


	#print "length of corpus is",len(corpus)

	printvectors = False
	if printvectors == True:
		i =0
		for doc in corpus_tfidf:
			print ("tfidf doc", confId[i], doc)
			i +=1

		i =0
		for doc in corpus_lsi:
			print ("lsi doc", confId[i], doc)
			i +=1
	####### not important

	#compute similarity of corpus against itself
	listofMethods = [ 'corpus_lsi', 'corpus_tfidf']
	for method in listofMethods:
		if method == 'corpus_lsi':
			cor = corpus_lsi
		elif method == 'corpus_tfidf':
			cor = corpus_tfidf

		index = similarities.MatrixSimilarity(cor)
		confSims = dict()
		confSimsDict = dict() # dictionary of [confId1][confId2]
		j=0
		sims = []
		for vec_tfidf in cor:
			sims = index[vec_tfidf]
			sims =sorted(enumerate(sims), key=lambda item: -item[1])
			confSims[confId[j]] = sims # in khat be dard nemikhore
			confSimsDict[j] = dict(sims)
			#print "index: ",confIdtoIndex[confId[j]], "confId: ", confId[j], confSims[confId[j]]
			j +=1

		if method == 'corpus_lsi':
			cslsi = dict()
			for c1index in confSimsDict.keys():
				cslsi[confId[c1index]] = dict()
				for c2index in confSimsDict.keys():
					cslsi[confId[c1index]][confId[c2index]] = confSimsDict[c1index][c2index]

		elif method == 'corpus_tfidf':
			cstfidf = dict()
			for c1index in confSimsDict.keys():
				cstfidf[confId[c1index]] = dict()
				for c2index in confSimsDict.keys():
					cstfidf[confId[c1index]][confId[c2index]] = confSimsDict[c1index][c2index]

	return cstfidf, cslsi

def start(path,outputDir):
	conf, author, term, year, conf_field, conf_paper, paper = parseDataset(path)
	confId, confIdtoIndex, corpus, dictionary = extractCorpusWithGensim(path, outputDir,conf,conf_paper, paper)
	cstfidf, cslsi = getTfidfLsiSims(corpus, confId, confIdtoIndex, dictionary, outputDir)
	return conf, cstfidf, cslsi
