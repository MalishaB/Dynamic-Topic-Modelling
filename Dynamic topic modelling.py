# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 17:15:58 2017

"""
import numpy as np 
import pandas as pd 
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import re
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim.models import ldamodel  
from gensim.models import coherencemodel 
from gensim.models import Phrases 
from gensim.models import ldaseqmodel    
import html 
from time import time
import seaborn as sns; sns.set()



#main data containing D1 only 
filename = r'data\ABCD.txt'

#------------------------1. CREATING DATA STRUCTURES----------------------------

documents = tuple(open(filename, 'r')) #list of tuples
df_data = pd.DataFrame(list(documents)) #dataframe
df_main = pd.DataFrame(df_data[0].str.split(' ',3).tolist(),
                                   columns = ['class','date','time','tweet'])#creating columns
df_main["rand"] = df_main["tweet"].str.slice(-20)#creating last rand column from tweet
df_main["tweet"] = df_main["tweet"].str.slice(2,-20)#removing rand num from tweet

#removing tweets labelled 0 - not cyber security labelled
df_main = df_main[df_main['class'] != '0']

#removing rows with ... as they are not complete tweets
df_main = df_main[df_main["tweet"].str.contains(r"\.\.\.") == False]

#----------------------2. CLEANING DATA---------------------------------------
#extracting number of docs in each month for LDASeq time slices
df_sorted = df_main.sort_values(by='date',ascending=True) #sort date
df_sorted = df_sorted.reset_index(drop=True) #reset index for sorted date
df_sorted['date'] =  pd.to_datetime(df_sorted['date'])
df_sorted['date'].index =  pd.to_datetime(df_sorted['date'].index) #change to date type
df_sorted['counter'] = 1 #add counter
per = df_sorted['date'].dt.to_period("M") #get freqs for each month
g = df_sorted.groupby(per)
df_mon = g.sum() #add to df
list_mon = df_mon['counter'].values.tolist() #extract freqs to list

doc_set = df_main["tweet"].tolist() #all tweets in a list for more preprocessing

tokenizer = RegexpTokenizer(r'\w+')
#stopwords = ['wordpress']
en_stop = get_stop_words('en')# create English stop words list
addedstop = ['can','now'] #add to list of stopwords
allstop = en_stop + addedstop
p_stemmer = PorterStemmer()# Create p_stemmer of class PorterStemmer
wordnet_lemmatizer = WordNetLemmatizer()

texts = []
for i in doc_set:  
    raw = re.sub(r"RT ", "",i) 
    raw = raw.lower()     
    clean = re.sub(r"(?:\@|https?\://)\S+", "",raw)#removing urls
    clean = re.sub(r"(\\x(.){2})", "",clean) #removing html code
    clean = html.unescape(clean)
    clean = re.sub(r"r/t ", "",clean)      
    clean = re.sub(r"\.", "",clean)
    clean = re.sub(r"-", "",clean)    
    clean = re.sub(r"_", "",clean)
    clean = re.sub(r"\(", "",clean)
    clean = re.sub(r"\)", "",clean)    
    clean = clean.replace(r"\n", "")
    clean = re.sub(r"(?:@[\w_]+)?", "",clean)
    clean = re.sub(r"<[^>]+>'", "",clean)
    #removing non-ascii characters
    clean = re.sub(r"vuln ", "vulnerability ",clean)
    clean = re.sub(r"mac os x", "macosx",clean)
    clean = re.sub(r"0day", "zeroday",clean)
    tokens = tokenizer.tokenize(clean)#tokenize 
    stopped_tokens = [i for i in tokens if not i in allstop] # remove stop words 
    lemmed_tokens = [wordnet_lemmatizer.lemmatize(i) for i in stopped_tokens] 
    #stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]# stem tokens
    texts.append(lemmed_tokens)   
    

#####creating bigrams and trigrams    
# Add bigrams and trigrams to docs default 10 time occurance 
#trigram added as replacment
bigram = Phrases(texts,min_count=10)
trigram = Phrases(bigram[texts],min_count=10)
for idx in range(len(texts)):
    texts[idx] = trigram[bigram[texts[idx]]]        

#trigram added as extra tokens
#for idx in range(len(textsTST)):
#    for token in trigram[bigram[textsTST[idx]]]:        
#        if '_' in token:
#            # Token is a bigram, add to document.
#            textsTST[idx].append(token)
      
                      
#-----------------------3. LDA TOPIC MODELLING--------------------------------  
dictionary = corpora.Dictionary(texts)# turn our tokenized documents into a id <-> term dictionary
corpus = [dictionary.doc2bow(text) for text in texts]# convert tokenized documents into a document-term matrix
print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))


#FINDIG BEST LIMIT FOR NUM OF TOPICS
def evaluate_graph(dictionary, corpus, texts, limit):
    c_v = []  
    lm_list = []  
    for num_topics in range(1, limit):
        lm = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        lm_list.append(lm)
        cm = gensim.models.coherencemodel.CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v.append(cm.get_coherence())
        
    # Show graph
    fig = plt.figure()
    x = range(1, limit)
    plt.plot(x, c_v)
    plt.xlabel("num_topics")
    plt.ylabel("coherence")
    plt.legend(("c_v"), loc='best')
    plt.show()
    fig.savefig('numtopicslimit.png') 
    
    return lm_list, c_v

start = time()
lmlist, c_v = evaluate_graph(dictionary=dictionary, corpus=corpus, texts=texts, limit=20)
print('Cell took %.2f seconds to run.' %(time() - start)) 

#PARAMETER SEARCH FOR BEST TUNED MODEL
start = time()
c_v = []  
lm_list = []  
topics = []
iters = []
chunks = []
numtop = 10
numiter = [1,10,50,100]
numchunk = [100,500,1000]
#ADD CHUNCKSIZE TO PARAM TESTING
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
for num_topics in range(3,numtop+1):
    for iter_val in numiter:
        for chunk_val in numchunk:
            lm = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, update_every=1, chunksize=chunk_val, passes= iter_val)
            lm_list.append(lm)
            cm = gensim.models.coherencemodel.CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
            c_v.append(cm.get_coherence())
            topics.append(num_topics)
            iters.append(iter_val)
            chunks.append(chunk_val)
            cohmatrix = pd.DataFrame({'coherence':c_v, 'ldamodelnum': lm_list, 'numtopics':topics, 'numpasses': iters, 'chunksize': chunks})
print('Cell took %.2f seconds to run.' %(time() - start))  

#exporting to csv
cohmatrix.to_csv('topics\cohmatrix.csv')
    
#checking best model
LDAmain = lm_list[30]
LDAmain.print_topics(num_words=20)
testoutput = LDAmain.print_topics(num_words=20)
LDAmain.save("models\ldamain")
#loading saved model to keep consistency
#fname = r"models\ldamain"
#LDAmain = gensim.models.ldamodel.LdaModel.load(fname, mmap='r')

#exporting output 
open("topics\output.txt", 'w').write('\n'.join('%s %s' % x for x in testoutput))

#create word clouds
for t in range(LDAmain.num_topics):
    fig = plt.figure()
    plt.imshow(WordCloud(background_color='white',colormap='brg',min_font_size=4,max_words=80, scale=1).fit_words(dict(LDAmain.show_topic(t, 200))))
    plt.axis("off")
    plt.title("Topic #" + str(t))
    plt.show()
    fig.savefig("visuals\LDAmaintop_%s.png" % t) 

    
#-----------------------4. Dynamic topic modelling-------------------------------
time_slice = list_mon #declaring time slice
ldaseq = ldaseqmodel.LdaSeqModel(initialize='ldamodel', lda_model=LDAmain, num_topics=5, corpus=corpus, id2word=dictionary, time_slice=time_slice,chain_variance=0.05)    
ldaseq.print_topics(time=0,top_terms=20)
ldaseq.print_topic_times(topic=0,top_terms=20) 
ldaseq.save("models\ldaseqmain")

#exporting results
np.savetxt("topics\dtmtop0file.csv", ldaseq.print_topic_times(topic=0), delimiter=",", fmt='%s')
np.savetxt("topics\dtmtop1file.csv", ldaseq.print_topic_times(topic=1), delimiter=",", fmt='%s')
np.savetxt("topics\dtmtop2file.csv", ldaseq.print_topic_times(topic=2), delimiter=",", fmt='%s')
np.savetxt("topics\dtmtop3file.csv", ldaseq.print_topic_times(topic=3), delimiter=",", fmt='%s')
np.savetxt("topics\dtmtop4file.csv", ldaseq.print_topic_times(topic=4), delimiter=",", fmt='%s')



#---------------------------Topic similarity-----------------------------------

#creating vec to be compared
lda_vec0 = sorted(LDAmain.get_topic_terms(0, topn=LDAmain.num_terms))
lda_vec1 = sorted(LDAmain.get_topic_terms(1, topn=LDAmain.num_terms))
lda_vec2 = sorted(LDAmain.get_topic_terms(2, topn=LDAmain.num_terms))
lda_vec3 = sorted(LDAmain.get_topic_terms(3, topn=LDAmain.num_terms))
lda_vec4 = sorted(LDAmain.get_topic_terms(4, topn=LDAmain.num_terms))

#hellinger calculaion for every combination
top00=gensim.matutils.hellinger(lda_vec0, lda_vec0)
top01=gensim.matutils.hellinger(lda_vec0, lda_vec1)
top02=gensim.matutils.hellinger(lda_vec0, lda_vec2)
top03=gensim.matutils.hellinger(lda_vec0, lda_vec3)
top04=gensim.matutils.hellinger(lda_vec0, lda_vec4)
top11=gensim.matutils.hellinger(lda_vec1, lda_vec1)
top12=gensim.matutils.hellinger(lda_vec1, lda_vec2)
top13=gensim.matutils.hellinger(lda_vec1, lda_vec3)
top14=gensim.matutils.hellinger(lda_vec1, lda_vec4)
top22=gensim.matutils.hellinger(lda_vec2, lda_vec2)
top23=gensim.matutils.hellinger(lda_vec2, lda_vec3)
top24=gensim.matutils.hellinger(lda_vec2, lda_vec4)
top33=gensim.matutils.hellinger(lda_vec3, lda_vec3)
top34=gensim.matutils.hellinger(lda_vec3, lda_vec4)
top44=gensim.matutils.hellinger(lda_vec4, lda_vec4)

#values to a dataframe
ldatopsim_df = pd.DataFrame([[top00, top01,top02, top03, top04], [top01,top11, top12,top13,top14], 
                             [top02,top12,top22,top23,top24],[top03,top13,top23,top33,top34],
                             [top04,top14,top24,top34,top44]], columns=list('01234'))

#heatmap and save
axall = sns.heatmap(ldatopsim_df,annot=True)
fig0 = axall.get_figure()
fig0.savefig("visuals\heatmapTopicALL.png")


#creating vec to be compared
ldaseq_vec0 = sorted(ldaseq.print_topics(time=0,top_terms=1748))
ldaseq_vec1 = sorted(ldaseq.print_topics(time=1,top_terms=1748))
ldaseq_vec2 = sorted(ldaseq.print_topics(time=2,top_terms=1748))
ldaseq_vec3 = sorted(ldaseq.print_topics(time=3,top_terms=1748))
ldaseq_vec4 = sorted(ldaseq.print_topics(time=4,top_terms=1748))
ldaseq_vec5 = sorted(ldaseq.print_topics(time=5,top_terms=1748))
ldaseq_vec6 = sorted(ldaseq.print_topics(time=6,top_terms=1748))

#getting vec in right shape for hellinger
topic0time0 = ldaseq_vec0[0] 
for i, (a, b) in enumerate(topic0time0):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic0time0[i] = (new_a, b)
     
topic1time0 = ldaseq_vec0[1]
for i, (a, b) in enumerate(topic1time0):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic1time0[i] = (new_a, b)
    
topic2time0 = ldaseq_vec0[2]
for i, (a, b) in enumerate(topic2time0):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic2time0[i] = (new_a, b)
    
topic3time0 = ldaseq_vec0[3]
for i, (a, b) in enumerate(topic3time0):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic3time0[i] = (new_a, b)
    
topic4time0 = ldaseq_vec0[4]
for i, (a, b) in enumerate(topic4time0):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic4time0[i] = (new_a, b)
    

topic0time1 = ldaseq_vec1[0]
for i, (a, b) in enumerate(topic0time1):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic0time1[i] = (new_a, b)

topic1time1 = ldaseq_vec1[1]
for i, (a, b) in enumerate(topic1time1):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic1time1[i] = (new_a, b)
    
topic2time1 = ldaseq_vec1[2]
for i, (a, b) in enumerate(topic2time1):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic2time1[i] = (new_a, b)
topic3time1 = ldaseq_vec1[3]
for i, (a, b) in enumerate(topic3time1):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic3time1[i] = (new_a, b)
topic4time1 = ldaseq_vec1[4]
for i, (a, b) in enumerate(topic4time1):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic4time1[i] = (new_a, b)

topic0time2 = ldaseq_vec2[0]
for i, (a, b) in enumerate(topic0time2):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic0time2[i] = (new_a, b)
    
topic1time2 = ldaseq_vec2[1]
for i, (a, b) in enumerate(topic1time2):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic1time2[i] = (new_a, b)    
    
topic2time2 = ldaseq_vec2[2]
for i, (a, b) in enumerate(topic2time2):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic2time2[i] = (new_a, b)
topic3time2 = ldaseq_vec2[3]
for i, (a, b) in enumerate(topic3time2):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic3time2[i] = (new_a, b)
topic4time2 = ldaseq_vec2[4]
for i, (a, b) in enumerate(topic4time2):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic4time2[i] = (new_a, b)

topic0time3 = ldaseq_vec3[0]
for i, (a, b) in enumerate(topic0time3):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic0time3[i] = (new_a, b)
topic1time3 = ldaseq_vec3[1]
for i, (a, b) in enumerate(topic1time3):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic1time3[i] = (new_a, b)
topic2time3 = ldaseq_vec3[2]
for i, (a, b) in enumerate(topic2time3):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic2time3[i] = (new_a, b)
topic3time3 = ldaseq_vec3[3]
for i, (a, b) in enumerate(topic3time3):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic3time3[i] = (new_a, b)
topic4time3 = ldaseq_vec3[4]
for i, (a, b) in enumerate(topic4time3):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic4time3[i] = (new_a, b)

topic0time4 = ldaseq_vec4[0]
for i, (a, b) in enumerate(topic0time4):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic0time4[i] = (new_a, b)
topic1time4 = ldaseq_vec4[1]
for i, (a, b) in enumerate(topic1time4):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic1time4[i] = (new_a, b)
topic2time4 = ldaseq_vec4[2]
for i, (a, b) in enumerate(topic2time4):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic2time4[i] = (new_a, b)
topic3time4 = ldaseq_vec4[3]
for i, (a, b) in enumerate(topic3time4):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic3time4[i] = (new_a, b)
topic4time4 = ldaseq_vec4[4]
for i, (a, b) in enumerate(topic4time4):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic4time4[i] = (new_a, b)

topic0time5 = ldaseq_vec5[0]
for i, (a, b) in enumerate(topic0time5):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic0time5[i] = (new_a, b)
topic1time5 = ldaseq_vec5[1]
for i, (a, b) in enumerate(topic1time5):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic1time5[i] = (new_a, b)
topic2time5 = ldaseq_vec5[2]
for i, (a, b) in enumerate(topic2time5):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic2time5[i] = (new_a, b)
    
topic3time5 = ldaseq_vec5[3]
for i, (a, b) in enumerate(topic3time5):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic3time5[i] = (new_a, b)
topic4time5 = ldaseq_vec5[4]
for i, (a, b) in enumerate(topic4time5):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic4time5[i] = (new_a, b)

topic0time6 = ldaseq_vec6[0]
for i, (a, b) in enumerate(topic0time6):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic0time6[i] = (new_a, b)   
    
topic1time6 = ldaseq_vec6[1]
for i, (a, b) in enumerate(topic1time6):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic1time6[i] = (new_a, b)
topic2time6 = ldaseq_vec6[2]
for i, (a, b) in enumerate(topic2time6):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic2time6[i] = (new_a, b)
topic3time6 = ldaseq_vec6[3]
for i, (a, b) in enumerate(topic3time6):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic3time6[i] = (new_a, b)
topic4time6 = ldaseq_vec6[4]
for i, (a, b) in enumerate(topic4time6):
    for idx in range(len(dictionary)):
        word = LDAmain.id2word[idx]
        if word == a:
            new_a = idx
    topic4time6[i] = (new_a, b)

#closer to 0 then more related, closer to 1 then less related
#extract data for each topic similarity for each time slice - heatmap for each time slice

topseq000=gensim.matutils.hellinger(topic0time0, topic0time0)
topseq001=gensim.matutils.hellinger(topic0time0, topic0time1)
topseq002=gensim.matutils.hellinger(topic0time0, topic0time2)
topseq003=gensim.matutils.hellinger(topic0time0, topic0time3)
topseq004=gensim.matutils.hellinger(topic0time0, topic0time4)
topseq005=gensim.matutils.hellinger(topic0time0, topic0time5)
topseq006=gensim.matutils.hellinger(topic0time0, topic0time6)
topseq011=gensim.matutils.hellinger(topic0time1, topic0time1)
topseq012=gensim.matutils.hellinger(topic0time1, topic0time2)
topseq013=gensim.matutils.hellinger(topic0time1, topic0time3)
topseq014=gensim.matutils.hellinger(topic0time1, topic0time4)
topseq015=gensim.matutils.hellinger(topic0time1, topic0time5)
topseq016=gensim.matutils.hellinger(topic0time1, topic0time6)
topseq022=gensim.matutils.hellinger(topic0time2, topic0time2)
topseq023=gensim.matutils.hellinger(topic0time2, topic0time3)
topseq024=gensim.matutils.hellinger(topic0time2, topic0time4)
topseq025=gensim.matutils.hellinger(topic0time2, topic0time5)
topseq026=gensim.matutils.hellinger(topic0time2, topic0time6)
topseq033=gensim.matutils.hellinger(topic0time3, topic0time3)
topseq034=gensim.matutils.hellinger(topic0time3, topic0time4)
topseq035=gensim.matutils.hellinger(topic0time3, topic0time5)
topseq036=gensim.matutils.hellinger(topic0time3, topic0time6)
topseq044=gensim.matutils.hellinger(topic0time4, topic0time4)
topseq045=gensim.matutils.hellinger(topic0time4, topic0time5)
topseq046=gensim.matutils.hellinger(topic0time4, topic0time6)
topseq055=gensim.matutils.hellinger(topic0time5, topic0time5)
topseq056=gensim.matutils.hellinger(topic0time5, topic0time6)
topseq066=gensim.matutils.hellinger(topic0time6, topic0time6) 

ldaseqtop0time06_df = pd.DataFrame([[topseq000,topseq001,topseq002,topseq003,topseq004,topseq005,topseq006], 
								[topseq001,topseq011,topseq012,topseq013,topseq014,topseq015,topseq016], 
								[topseq002,topseq012,topseq022,topseq023,topseq024,topseq025,topseq026],
								[topseq003,topseq013,topseq023,topseq033,topseq034,topseq035,topseq036],
								[topseq004,topseq014,topseq024,topseq034,topseq044,topseq045,topseq046],
								[topseq005,topseq015,topseq025,topseq035,topseq045,topseq055,topseq056],
								[topseq006,topseq016,topseq026,topseq036,topseq046,topseq056,topseq066]], columns=list('0123456'))

ax0 = sns.heatmap(ldaseqtop0time06_df,annot=True)
fig0 = ax0.get_figure()
fig0.savefig("visuals\heatmapTopic0.png")

topseq100=gensim.matutils.hellinger(topic1time0, topic1time0)
topseq101=gensim.matutils.hellinger(topic1time0, topic1time1)
topseq102=gensim.matutils.hellinger(topic1time0, topic1time2)
topseq103=gensim.matutils.hellinger(topic1time0, topic1time3)
topseq104=gensim.matutils.hellinger(topic1time0, topic1time4)
topseq105=gensim.matutils.hellinger(topic1time0, topic1time5)
topseq106=gensim.matutils.hellinger(topic1time0, topic1time6)
topseq111=gensim.matutils.hellinger(topic1time1, topic1time1)
topseq112=gensim.matutils.hellinger(topic1time1, topic1time2)
topseq113=gensim.matutils.hellinger(topic1time1, topic1time3)
topseq114=gensim.matutils.hellinger(topic1time1, topic1time4)
topseq115=gensim.matutils.hellinger(topic1time1, topic1time5)
topseq116=gensim.matutils.hellinger(topic1time1, topic1time6)
topseq122=gensim.matutils.hellinger(topic1time2, topic1time2)
topseq123=gensim.matutils.hellinger(topic1time2, topic1time3)
topseq124=gensim.matutils.hellinger(topic1time2, topic1time4)
topseq125=gensim.matutils.hellinger(topic1time2, topic1time5)
topseq126=gensim.matutils.hellinger(topic1time2, topic1time6)
topseq133=gensim.matutils.hellinger(topic1time3, topic1time3)
topseq134=gensim.matutils.hellinger(topic1time3, topic1time4)
topseq135=gensim.matutils.hellinger(topic1time3, topic1time5)
topseq136=gensim.matutils.hellinger(topic1time3, topic1time6)
topseq144=gensim.matutils.hellinger(topic1time4, topic1time4)
topseq145=gensim.matutils.hellinger(topic1time4, topic1time5)
topseq146=gensim.matutils.hellinger(topic1time4, topic1time6)
topseq155=gensim.matutils.hellinger(topic1time5, topic1time5)
topseq156=gensim.matutils.hellinger(topic1time5, topic1time6)
topseq166=gensim.matutils.hellinger(topic1time6, topic1time6) 

ldaseqtop1time06_df = pd.DataFrame([[topseq100,topseq101,topseq102,topseq103,topseq104,topseq105,topseq106], 
									[topseq101,topseq111,topseq112,topseq113,topseq114,topseq115,topseq116], 
									[topseq102,topseq112,topseq122,topseq123,topseq124,topseq125,topseq126],
									[topseq103,topseq113,topseq123,topseq133,topseq134,topseq135,topseq136],
									[topseq104,topseq114,topseq124,topseq134,topseq144,topseq145,topseq146],
									[topseq105,topseq115,topseq125,topseq135,topseq145,topseq155,topseq156],
									[topseq106,topseq116,topseq126,topseq136,topseq146,topseq156,topseq166]], columns=list('0123456'))

ax1 = sns.heatmap(ldaseqtop1time06_df,annot=True)
fig1 = ax1.get_figure()
fig1.savefig("visuals\heatmapTopic1.png")

topseq200=gensim.matutils.hellinger(topic2time0, topic2time0)
topseq201=gensim.matutils.hellinger(topic2time0, topic2time1)
topseq202=gensim.matutils.hellinger(topic2time0, topic2time2)
topseq203=gensim.matutils.hellinger(topic2time0, topic2time3)
topseq204=gensim.matutils.hellinger(topic2time0, topic2time4)
topseq205=gensim.matutils.hellinger(topic2time0, topic2time5)
topseq206=gensim.matutils.hellinger(topic2time0, topic2time6)
topseq211=gensim.matutils.hellinger(topic2time1, topic2time1)
topseq212=gensim.matutils.hellinger(topic2time1, topic2time2)
topseq213=gensim.matutils.hellinger(topic2time1, topic2time3)
topseq214=gensim.matutils.hellinger(topic2time1, topic2time4)
topseq215=gensim.matutils.hellinger(topic2time1, topic2time5)
topseq216=gensim.matutils.hellinger(topic2time1, topic2time6)
topseq222=gensim.matutils.hellinger(topic2time2, topic2time2)
topseq223=gensim.matutils.hellinger(topic2time2, topic2time3)
topseq224=gensim.matutils.hellinger(topic2time2, topic2time4)
topseq225=gensim.matutils.hellinger(topic2time2, topic2time5)
topseq226=gensim.matutils.hellinger(topic2time2, topic2time6)
topseq233=gensim.matutils.hellinger(topic2time3, topic2time3)
topseq234=gensim.matutils.hellinger(topic2time3, topic2time4)
topseq235=gensim.matutils.hellinger(topic2time3, topic2time5)
topseq236=gensim.matutils.hellinger(topic2time3, topic2time6)
topseq244=gensim.matutils.hellinger(topic2time4, topic2time4)
topseq245=gensim.matutils.hellinger(topic2time4, topic2time5)
topseq246=gensim.matutils.hellinger(topic2time4, topic2time6)
topseq255=gensim.matutils.hellinger(topic2time5, topic2time5)
topseq256=gensim.matutils.hellinger(topic2time5, topic2time6)
topseq266=gensim.matutils.hellinger(topic2time6, topic2time6) 

ldaseqtop2time06_df = pd.DataFrame([[topseq200,topseq201,topseq202,topseq203,topseq204,topseq205,topseq206], 
									[topseq201,topseq211,topseq212,topseq213,topseq214,topseq215,topseq216], 
									[topseq202,topseq212,topseq222,topseq223,topseq224,topseq225,topseq226],
									[topseq203,topseq213,topseq223,topseq233,topseq234,topseq235,topseq236],
									[topseq204,topseq214,topseq224,topseq234,topseq244,topseq245,topseq246],
									[topseq205,topseq215,topseq225,topseq235,topseq245,topseq255,topseq256],
									[topseq206,topseq216,topseq226,topseq236,topseq246,topseq256,topseq266]], columns=list('0123456'))

ax2 = sns.heatmap(ldaseqtop2time06_df,annot=True)
fig2 = ax2.get_figure()
fig2.savefig("visuals\heatmapTopic2.png")

topseq300=gensim.matutils.hellinger(topic3time0, topic3time0)
topseq301=gensim.matutils.hellinger(topic3time0, topic3time1)
topseq302=gensim.matutils.hellinger(topic3time0, topic3time2)
topseq303=gensim.matutils.hellinger(topic3time0, topic3time3)
topseq304=gensim.matutils.hellinger(topic3time0, topic3time4)
topseq305=gensim.matutils.hellinger(topic3time0, topic3time5)
topseq306=gensim.matutils.hellinger(topic3time0, topic3time6)
topseq311=gensim.matutils.hellinger(topic3time1, topic3time1)
topseq312=gensim.matutils.hellinger(topic3time1, topic3time2)
topseq313=gensim.matutils.hellinger(topic3time1, topic3time3)
topseq314=gensim.matutils.hellinger(topic3time1, topic3time4)
topseq315=gensim.matutils.hellinger(topic3time1, topic3time5)
topseq316=gensim.matutils.hellinger(topic3time1, topic3time6)
topseq322=gensim.matutils.hellinger(topic3time2, topic3time2)
topseq323=gensim.matutils.hellinger(topic3time2, topic3time3)
topseq324=gensim.matutils.hellinger(topic3time2, topic3time4)
topseq325=gensim.matutils.hellinger(topic3time2, topic3time5)
topseq326=gensim.matutils.hellinger(topic3time2, topic3time6)
topseq333=gensim.matutils.hellinger(topic3time3, topic3time3)
topseq334=gensim.matutils.hellinger(topic3time3, topic3time4)
topseq335=gensim.matutils.hellinger(topic3time3, topic3time5)
topseq336=gensim.matutils.hellinger(topic3time3, topic3time6)
topseq344=gensim.matutils.hellinger(topic3time4, topic3time4)
topseq345=gensim.matutils.hellinger(topic3time4, topic3time5)
topseq346=gensim.matutils.hellinger(topic3time4, topic3time6)
topseq355=gensim.matutils.hellinger(topic3time5, topic3time5)
topseq356=gensim.matutils.hellinger(topic3time5, topic3time6)
topseq366=gensim.matutils.hellinger(topic3time6, topic3time6) 

ldaseqtop3time06_df = pd.DataFrame([[topseq300,topseq301,topseq302,topseq303,topseq304,topseq305,topseq306], 
									[topseq301,topseq311,topseq312,topseq313,topseq314,topseq315,topseq316], 
									[topseq302,topseq312,topseq322,topseq323,topseq324,topseq325,topseq326],
									[topseq303,topseq313,topseq323,topseq333,topseq334,topseq335,topseq336],
									[topseq304,topseq314,topseq324,topseq334,topseq344,topseq345,topseq346],
									[topseq305,topseq315,topseq325,topseq335,topseq345,topseq355,topseq356],
									[topseq306,topseq316,topseq326,topseq336,topseq346,topseq356,topseq366]], columns=list('0123456'))

ax3 = sns.heatmap(ldaseqtop3time06_df,annot=True)
fig3 = ax3.get_figure()
fig3.savefig("visuals\heatmapTopic3.png")

topseq400=gensim.matutils.hellinger(topic4time0, topic4time0)
topseq401=gensim.matutils.hellinger(topic4time0, topic4time1)
topseq402=gensim.matutils.hellinger(topic4time0, topic4time2)
topseq403=gensim.matutils.hellinger(topic4time0, topic4time3)
topseq404=gensim.matutils.hellinger(topic4time0, topic4time4)
topseq405=gensim.matutils.hellinger(topic4time0, topic4time5)
topseq406=gensim.matutils.hellinger(topic4time0, topic4time6)
topseq411=gensim.matutils.hellinger(topic4time1, topic4time1)
topseq412=gensim.matutils.hellinger(topic4time1, topic4time2)
topseq413=gensim.matutils.hellinger(topic4time1, topic4time3)
topseq414=gensim.matutils.hellinger(topic4time1, topic4time4)
topseq415=gensim.matutils.hellinger(topic4time1, topic4time5)
topseq416=gensim.matutils.hellinger(topic4time1, topic4time6)
topseq422=gensim.matutils.hellinger(topic4time2, topic4time2)
topseq423=gensim.matutils.hellinger(topic4time2, topic4time3)
topseq424=gensim.matutils.hellinger(topic4time2, topic4time4)
topseq425=gensim.matutils.hellinger(topic4time2, topic4time5)
topseq426=gensim.matutils.hellinger(topic4time2, topic4time6)
topseq433=gensim.matutils.hellinger(topic4time3, topic4time3)
topseq434=gensim.matutils.hellinger(topic4time3, topic4time4)
topseq435=gensim.matutils.hellinger(topic4time3, topic4time5)
topseq436=gensim.matutils.hellinger(topic4time3, topic4time6)
topseq444=gensim.matutils.hellinger(topic4time4, topic4time4)
topseq445=gensim.matutils.hellinger(topic4time4, topic4time5)
topseq446=gensim.matutils.hellinger(topic4time4, topic4time6)
topseq455=gensim.matutils.hellinger(topic4time5, topic4time5)
topseq456=gensim.matutils.hellinger(topic4time5, topic4time6)
topseq466=gensim.matutils.hellinger(topic4time6, topic4time6) 

ldaseqtop4time06_df = pd.DataFrame([[topseq400,topseq401,topseq402,topseq403,topseq404,topseq405,topseq406], 
									[topseq401,topseq411,topseq412,topseq413,topseq414,topseq415,topseq416], 
									[topseq402,topseq412,topseq422,topseq423,topseq424,topseq425,topseq426],
									[topseq403,topseq413,topseq423,topseq433,topseq434,topseq435,topseq436],
									[topseq404,topseq414,topseq424,topseq434,topseq444,topseq445,topseq446],
									[topseq405,topseq415,topseq425,topseq435,topseq445,topseq455,topseq456],
									[topseq406,topseq416,topseq426,topseq436,topseq446,topseq456,topseq466]], columns=list('0123456'))

ax4 = sns.heatmap(ldaseqtop4time06_df,annot=True)
fig4 = ax4.get_figure()
fig4.savefig("visuals\heatmapTopic4.png")





  