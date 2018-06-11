# pip install -U textblob
# python -m textblob.download_corpora

import json
from collections import defaultdict 
from textblob import TextBlob
import re

#############################################################################
# Step1 : data preprocessing


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r" v", " very", phrase)
    return phrase

reviewerID =[]
productID = []
reviewerName=[]
liked_and_seen = []
reviewText = []
rating = []
summary = []
unixTime = []
date = []

with open('Cell_Phones_and_Accessories_5.json') as json_data:
    d = json.load(json_data)
    d = d[0:300]

for i in range(len(d)):
    reviewText.append(decontracted(d[i]['reviewText']))
    rating.append(d[i]['overall'])
    reviewerID.append(d[i]['reviewerID'])
    productID.append(d[i]['asin'])
#    reviewerName.append(d[i]['reviewerName'])
    liked_and_seen.append(d[i]['helpful'])
    summary.append(d[i]['summary'])
    unixTime.append(d[i]['unixReviewTime'])
    date.append(d[i]['reviewTime'])
    

# create  dataset
from pandas import DataFrame
dataset = DataFrame({'reviewerID': reviewerID, 'productID': productID, 'liked_and_seen': liked_and_seen, 'reviewText': reviewText, 'rating': rating, 'summary': summary, 'unixTime': unixTime, 'date': date})

#cleaning unwanted symbols
#cleaning unwanted symbols
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

comment_dict = defaultdict(list)
for i in range(len(dataset)):
    sentence = re.sub('[^a-zA-Z.]',' ',dataset['reviewText'][i])
    sentence = sentence.lower()
    sentence = sentence.split('.')
    for k in range(len(sentence)):
        review = sentence[k].split()
        review = [word for word in review if not word in set(stopwords.words('english'))]
        sentence[k] =  ' '.join(review)
        comment_dict[i].append(sentence[k])
 
#delete unwanted '' words
for j in range(len(comment_dict)):
    comment_dict[j] = [comment_dict[j][i] for i in range(len(comment_dict[j])) if comment_dict[j][i] not in '']

for i in range(len(comment_dict)):
    reviewText[i] = ('. '.join(comment_dict[i][j] for j in range(len(comment_dict[i]))))

# spelling correction
for i in range(len(reviewText)):
    b = TextBlob(reviewText[i])
    reviewText[i] = b.correct()
    
dataset_corrected= DataFrame({'reviewerID': reviewerID, 'productID': productID, 'liked_and_seen': liked_and_seen, 'reviewText': reviewText, 'rating': rating, 'summary': summary, 'unixTime': unixTime, 'date': date})
 
# creating corpus
corpus = defaultdict(set)
for i in range(len(reviewText)):
    wiki = reviewText[i]
    corpus[i] = wiki.sentences
            
corpus_key = corpus.keys()
corpus_list = defaultdict(list)

for i in corpus_key:
    for j in range(len(corpus[i])):
        word = ' '.join(corpus[i][j].words)
        corpus_list[i].append(word)

####################################################################################    
# Step 2: adding biwords and triwords for generating patterns

length = defaultdict(list)
for i in corpus_key:
    length[i] = list(corpus_list[i])

# triwords       
for i in corpus_key:
    for j in range(len(length[i])):
        text = TextBlob(length[i][j])
        text = text.ngrams(n=3)
        for k in range(len(text)):
            triword = [' '.join([text[k][l] for l in range(len(text[k]))])]
            triword = triword[0]
            corpus_list[i].append(triword)
#biwords
for i in corpus_key:
    for j in range(len(length[i])):
        text = TextBlob(length[i][j])
        text = text.ngrams(n=2)
        for k in range(len(text)):
            triword = [' '.join([text[k][l] for l in range(len(text[k]))])]
            triword = triword[0]
            corpus_list[i].append(triword)
                  
    
# corpus_list creates a list of all Opinion Words present in review text
# here the corpus contains sentences, biwords and triwords from review 
# Also here only matched text is used
####################################################################################    
# Step 3: Part-of-speech Tagging 

pos_dict = defaultdict(list)
for i in corpus_key:
    for j in range(len(corpus_list[i])):
        text = TextBlob(corpus_list[i][j])
        text = text.tags
        pos_dict[i].append(text)

pos_dict_key = pos_dict.keys()

corpus_noun = defaultdict(list)
for i in pos_dict_key:
    for j in range(len(pos_dict[i])):
        for k in range(len(pos_dict[i][j])):
            if(pos_dict[i][j][k][1] == 'NN'):
                corpus_noun[i].append(pos_dict[i][j][k])          

##########################################################################################
# Step 4: pattern generation from Part -of -speech tagging
pattern1 =  defaultdict(list)
for i in pos_dict_key:
    for j in range(len(pos_dict[i])):
        if(len(pos_dict[i][j]) == 2):
            if((pos_dict[i][j][0][1] == 'JJ' and pos_dict[i][j][1][1] == 'NN') or (pos_dict[i][j][0][1] == 'JJ' and pos_dict[i][j][1][1] == 'NNS')):
                #pattern1  
                pattern1[i].append(pos_dict[i][j])
                                       
pattern2 =  defaultdict(list)
for i in pos_dict_key:
    for j in range(len(pos_dict[i])):          
        if(len(pos_dict[i][j]) == 3):
            if((pos_dict[i][j][0][1] == 'JJ' and pos_dict[i][j][1][1] == 'NN'and pos_dict[i][j][2][1] == 'NN') or (pos_dict[i][j][0][1] == 'JJ' and pos_dict[i][j][1][1] == 'NN'and pos_dict[i][j][2][1] == 'NNS')):
                #pattern2
                pattern2[i].append(pos_dict[i][j])
            elif((pos_dict[i][j][0][1] == 'JJ' and pos_dict[i][j][1][1] == 'NNS'and pos_dict[i][j][2][1] == 'NN') or (pos_dict[i][j][0][1] == 'JJ' and pos_dict[i][j][1][1] == 'NNS'and pos_dict[i][j][2][1] == 'NNS')):
                #pattern2
                pattern2[i].append(pos_dict[i][j])
                   
pattern3 =  defaultdict(list)
for i in pos_dict_key:
    for j in range(len(pos_dict[i])):
        if(len(pos_dict[i][j]) == 2):
            if((pos_dict[i][j][0][1] == 'RB' and pos_dict[i][j][1][1] == 'JJ') or (pos_dict[i][j][0][1] == 'RBR' and pos_dict[i][j][1][1] == 'JJ') or (pos_dict[i][j][0][1] == 'RBS' and pos_dict[i][j][1][1] == 'JJ')):
                #pattern3
                pattern3[i].append(pos_dict[i][j]) 

pattern4 =  defaultdict(list)
for i in pos_dict_key:
    for j in range(len(pos_dict[i])):          
        if(len(pos_dict[i][j]) == 3):
            if((pos_dict[i][j][0][1] == 'RB' and pos_dict[i][j][1][1] == 'JJ'and pos_dict[i][j][2][1] == 'NN') or (pos_dict[i][j][0][1] == 'RB' and pos_dict[i][j][1][1] == 'JJ'and pos_dict[i][j][2][1] == 'NNS')):
                #pattern4
                pattern4[i].append(pos_dict[i][j])
            elif((pos_dict[i][j][0][1] == 'RB' and pos_dict[i][j][1][1] == 'RB'and pos_dict[i][j][2][1] == 'NN') or (pos_dict[i][j][0][1] == 'RB' and pos_dict[i][j][1][1] == 'RB'and pos_dict[i][j][2][1] == 'NNS')):
                #pattern4
                pattern4[i].append(pos_dict[i][j])
            elif((pos_dict[i][j][0][1] == 'RB' and pos_dict[i][j][1][1] == 'RBR'and pos_dict[i][j][2][1] == 'NN') or (pos_dict[i][j][0][1] == 'RB' and pos_dict[i][j][1][1] == 'RBR'and pos_dict[i][j][2][1] == 'NNS')):
                #pattern4
                pattern4[i].append(pos_dict[i][j])
            elif((pos_dict[i][j][0][1] == 'RB' and pos_dict[i][j][1][1] == 'RBS'and pos_dict[i][j][2][1] == 'NN') or (pos_dict[i][j][0][1] == 'RB' and pos_dict[i][j][1][1] == 'RBS'and pos_dict[i][j][2][1] == 'NNS')):
                #pattern4
                pattern4[i].append(pos_dict[i][j])
            elif((pos_dict[i][j][0][1] == 'RBR' and pos_dict[i][j][1][1] == 'JJ'and pos_dict[i][j][2][1] == 'NN') or (pos_dict[i][j][0][1] == 'RB' and pos_dict[i][j][1][1] == 'JJ'and pos_dict[i][j][2][1] == 'NNS')):
                #pattern4
                pattern4[i].append(pos_dict[i][j])
            elif((pos_dict[i][j][0][1] == 'RBR' and pos_dict[i][j][1][1] == 'RB'and pos_dict[i][j][2][1] == 'NN') or (pos_dict[i][j][0][1] == 'RB' and pos_dict[i][j][1][1] == 'RB'and pos_dict[i][j][2][1] == 'NNS')):
                #pattern4
                pattern4[i].append(pos_dict[i][j])
            elif((pos_dict[i][j][0][1] == 'RBR' and pos_dict[i][j][1][1] == 'RBR'and pos_dict[i][j][2][1] == 'NN') or (pos_dict[i][j][0][1] == 'RB' and pos_dict[i][j][1][1] == 'RBR'and pos_dict[i][j][2][1] == 'NNS')):
                #pattern4
                pattern4[i].append(pos_dict[i][j])
            elif((pos_dict[i][j][0][1] == 'RBR' and pos_dict[i][j][1][1] == 'RBS'and pos_dict[i][j][2][1] == 'NN') or (pos_dict[i][j][0][1] == 'RB' and pos_dict[i][j][1][1] == 'RBS'and pos_dict[i][j][2][1] == 'NNS')):
                #pattern4
                pattern4[i].append(pos_dict[i][j])
            elif((pos_dict[i][j][0][1] == 'RBS' and pos_dict[i][j][1][1] == 'JJ'and pos_dict[i][j][2][1] == 'NN') or (pos_dict[i][j][0][1] == 'RB' and pos_dict[i][j][1][1] == 'JJ'and pos_dict[i][j][2][1] == 'NNS')):
                #pattern4
                pattern4[i].append(pos_dict[i][j])
            elif((pos_dict[i][j][0][1] == 'RBS' and pos_dict[i][j][1][1] == 'RB'and pos_dict[i][j][2][1] == 'NN') or (pos_dict[i][j][0][1] == 'RB' and pos_dict[i][j][1][1] == 'RB'and pos_dict[i][j][2][1] == 'NNS')):
                #pattern4
                pattern4[i].append(pos_dict[i][j])
            elif((pos_dict[i][j][0][1] == 'RBS' and pos_dict[i][j][1][1] == 'RBR'and pos_dict[i][j][2][1] == 'NN') or (pos_dict[i][j][0][1] == 'RB' and pos_dict[i][j][1][1] == 'RBR'and pos_dict[i][j][2][1] == 'NNS')):
                #pattern4
                pattern4[i].append(pos_dict[i][j])
            elif((pos_dict[i][j][0][1] == 'RBS' and pos_dict[i][j][1][1] == 'RBS'and pos_dict[i][j][2][1] == 'NN') or (pos_dict[i][j][0][1] == 'RB' and pos_dict[i][j][1][1] == 'RBS'and pos_dict[i][j][2][1] == 'NNS')):
                #pattern4
                pattern4[i].append(pos_dict[i][j])
                    
pattern5 =  defaultdict(list)
for i in pos_dict_key:
    for j in range(len(pos_dict[i])):
        if(len(pos_dict[i][j]) == 2):
            if((pos_dict[i][j][0][1] == 'RB' and pos_dict[i][j][1][1] == 'VBN') or (pos_dict[i][j][0][1] == 'RBR' and pos_dict[i][j][1][1] == 'VBN') or (pos_dict[i][j][0][1] == 'RBS' and pos_dict[i][j][1][1] == 'VBN')):
                #pattern5
                pattern5[i].append(pos_dict[i][j])
            elif((pos_dict[i][j][0][1] == 'RB' and pos_dict[i][j][1][1] == 'VBD') or (pos_dict[i][j][0][1] == 'RBR' and pos_dict[i][j][1][1] == 'VBD') or (pos_dict[i][j][0][1] == 'RBS' and pos_dict[i][j][1][1] == 'VBD')):
                #pattern5
                pattern5[i].append(pos_dict[i][j])
                
pattern6 =  defaultdict(list)
for i in pos_dict_key:
    for j in range(len(pos_dict[i])):          
        if(len(pos_dict[i][j]) == 3):
            if((pos_dict[i][j][0][1] == 'RB' and pos_dict[i][j][1][1] == 'RB'and pos_dict[i][j][2][1] == 'JJ') or (pos_dict[i][j][0][1] == 'RB' and pos_dict[i][j][1][1] == 'RBR'and pos_dict[i][j][2][1] == 'JJ') or (pos_dict[i][j][0][1] == 'RB' and pos_dict[i][j][1][1] == 'RBS'and pos_dict[i][j][0][1] == 'JJ')):
                #pattern6
                pattern6[i].append(pos_dict[i][j])
            elif((pos_dict[i][j][0][1] == 'RBR' and pos_dict[i][j][1][1] == 'RB'and pos_dict[i][j][2][1] == 'JJ') or (pos_dict[i][j][0][1] == 'RBR' and pos_dict[i][j][1][1] == 'RBR'and pos_dict[i][j][2][1] == 'JJ') or (pos_dict[i][j][0][1] == 'RBR' and pos_dict[i][j][1][1] == 'RBS'and pos_dict[i][j][0][1] == 'JJ')):
                #pattern6
                pattern6[i].append(pos_dict[i][j])
            elif((pos_dict[i][j][0][1] == 'RBS' and pos_dict[i][j][1][1] == 'RB'and pos_dict[i][j][2][1] == 'JJ') or (pos_dict[i][j][0][1] == 'RBS' and pos_dict[i][j][1][1] == 'RBR'and pos_dict[i][j][2][1] == 'JJ') or (pos_dict[i][j][0][1] == 'RBS' and pos_dict[i][j][1][1] == 'RBS'and pos_dict[i][j][0][1] == 'JJ')):
                #pattern6
                pattern6[i].append(pos_dict[i][j])

pattern7 =  defaultdict(list)
for i in pos_dict_key:
    for j in range(len(pos_dict[i])):
        if(len(pos_dict[i][j]) == 2):
            if((pos_dict[i][j][0][1] == 'VBN' and pos_dict[i][j][1][1] == 'NN') or (pos_dict[i][j][0][1] == 'VBD' and pos_dict[i][j][1][1] == 'NN')):
                #pattern7
                pattern7[i].append(pos_dict[i][j])
            elif((pos_dict[i][j][0][1] == 'VBN' and pos_dict[i][j][1][1] == 'NNS') or (pos_dict[i][j][0][1] == 'VBD' and pos_dict[i][j][1][1] == 'NNS')):
                #pattern7
                pattern7[i].append(pos_dict[i][j])

pattern8 =  defaultdict(list)
for i in pos_dict_key:
    for j in range(len(pos_dict[i])):
        if(len(pos_dict[i][j]) == 2):
            if((pos_dict[i][j][0][1] == 'VBN' and pos_dict[i][j][1][1] == 'RB') or (pos_dict[i][j][0][1] == 'VBD' and pos_dict[i][j][1][1] == 'RB')):
                #pattern8
                pattern8[i].append(pos_dict[i][j])
            elif((pos_dict[i][j][0][1] == 'VBN' and pos_dict[i][j][1][1] == 'RBR') or (pos_dict[i][j][0][1] == 'VBD' and pos_dict[i][j][1][1] == 'RBR')):
                #pattern8
                pattern8[i].append(pos_dict[i][j])
            elif((pos_dict[i][j][0][1] == 'VBN' and pos_dict[i][j][1][1] == 'RBS') or (pos_dict[i][j][0][1] == 'VBD' and pos_dict[i][j][1][1] == 'RBS')):
                #pattern8
                pattern8[i].append(pos_dict[i][j])

pattern = defaultdict(set)
pattern.update(pattern1)
pattern.update(pattern2)
pattern.update(pattern3)
pattern.update(pattern4)
pattern.update(pattern5)
pattern.update(pattern6)
pattern.update(pattern7)
pattern.update(pattern8)

#select stuff from OT_OW_key
####################################################################################
# Step 5: Semi- supervised approach creates Opinion Target from stuff.

stuff = ['software', 'application', 'service', 'power supply', 'sim card', 'display', 
         'storage space', 'sensor', 'wireless charging', 'design', 'cpu', 'accessories',
         'camera','quality','time','condition','screen','price','case','build','access',
         'battery','buy','power','switch','light','design','technology','radio','fashion'
'product','charging','feature','touch','profile','car','slot','tables','construction',
'period ','system','game','bottom','sound','blackberry charge','price anyone','price extra',
'cord length','charge port',' phone','horizon charge','fraction price','charge ','key',
'extension','internet','cheap','cover','speaker']

####################################################################################
# Step 6: Finding Similar waords of above Opinion Targets

from nltk.corpus import wordnet as wn
from itertools import chain

stuff_OT = defaultdict(set)
OT2 = set()

synsets_set = defaultdict(set)
hyponyms_set = defaultdict(set)
for z in range(len(stuff)): 
    input_word = stuff[z] 
    OT1= set()    
    for i,j in enumerate(wn.synsets(input_word)):
    #    print ('Meaning', i, 'NLTK ID: ', j.name())
        
        hypernyms = ', '.join(list(chain(*[l.lemma_names() for l in j.hypernyms()])))
    #    print ('Hypernyms:', hypernyms)
        synsets_set[i].add(hypernyms)
        
        hyponyms = ', '.join(list(chain(*[l.lemma_names() for l in j.hyponyms()])))
    #    print ('Hyponyms:', hyponyms)
        hyponyms_set[i].add(hyponyms)
    #    print() 

        ho = [hypernyms]
        for h in range(len(ho)):
            temp_list = ho[h].split(', ')
            if(temp_list != ['']):
                for l in range(len(temp_list)):
                    temp_word = ' '.join(temp_list[l].split('_'))
                    OT2.add(temp_word)
                    OT1.add(temp_word)
        
        hy = [hypernyms]
        for h in range(len(hy)):
            temp_list = hy[h].split(', ')
            if(temp_list != ['']):
                for l in range(len(temp_list)):
                    temp_word = ' '.join(temp_list[l].split('_'))
                    OT2.add(temp_word)
                    OT1.add(temp_word)
    OT1 = list(OT1)
    for i in range(len(OT1)):
        stuff_OT[stuff[z]].add(OT1[i])

# stuff contains Opinion Target and its similar words from wordnet        
####################################################################################

hyponyms_set_keys = hyponyms_set.keys()
synsets_set_keys = synsets_set.keys()

for z in range(len(stuff)):  
    
    for k in hyponyms_set_keys:
        hy = hyponyms_set[k]
        hy = list(hy)
        for h in range(len(hy)):
            temp_list = hy[h].split(', ')
            if(temp_list != ['']):
                for l in range(len(temp_list)):
                    temp_word = ' '.join(temp_list[l].split('_'))
                    OT2.add(temp_word)
                    
for z in range(len(stuff)):    
    for k in synsets_set_keys:
        hy = synsets_set[k]
        hy = list(hy)
        for h in range(len(hy)):
            temp_list = hy[h].split(', ')
            if(temp_list != ['']):
                for l in range(len(temp_list)):
                    temp_word = ' '.join(temp_list[l].split('_'))
                    OT2.add(temp_word)

OT2 = list(OT2)

list_of_subset = defaultdict(set)
for i in range(len(stuff)):
    stuff_OT[stuff[i]] = list(stuff_OT[stuff[i]])
    for j in range(len(stuff_OT[stuff[i]])):
        list_of_subset[i].add(stuff[i])
        list_of_subset[i].add(stuff_OT[stuff[i]][j])

for i in list_of_subset.keys():
    list_of_subset[i] = list(list_of_subset[i])
    
    
    

list_of_subset2 =[]
for i in range(len(stuff)):
    list_of_subset2.append(stuff[i])

for i in list_of_subset.keys():
    for j in range(len(list_of_subset[i])):
        list_of_subset2.append(list_of_subset[i][j])

stuff = list_of_subset2
#######################################################################################
# Step 7: Finding Opinion Words of above Opinion Target(Stuff + its similar words.)
#         from pattern generated in step 3.

OW = defaultdict(set)
OT = defaultdict(set)
OT_OW = defaultdict(set)
OW_OT = defaultdict(set)

# it has 1 OW
pattern1_OT_OW =  defaultdict(set)
pattern1_OW_OT =  defaultdict(set)
p1_keys = pattern1.keys()
for i in p1_keys:
    if( pattern1[i] != []):
        for j in range(len(pattern1[i])):
            OT[i].add(pattern1[i][j][1][0])
            OW[i].add(pattern1[i][j][0][0])
            if(pattern1[i][j][1][0] in stuff):
                OT_OW[pattern1[i][j][1][0]].add(pattern1[i][j][0][0])
                OW_OT[pattern1[i][j][0][0]].add(pattern1[i][j][1][0])
                
                pattern1_OT_OW[pattern1[i][j][1][0]].add(pattern1[i][j][0][0])
                pattern1_OW_OT[pattern1[i][j][0][0]].add(pattern1[i][j][1][0])

#it has 1 OW             
pattern2_OT_OW = defaultdict(set)
pattern2_OW_OT = defaultdict(set)
p2_keys = pattern2.keys()
for i in p2_keys:
    if( pattern2[i] != []):
        for j in range(len(pattern2[i])):    
            target = pattern2[i][j][1][0] + " " + pattern2[i][j][2][0]
            OT[i].add(target)
            OW[i].add(pattern2[i][j][0][0])
            if(pattern2[i][j][1][0] in stuff or pattern2[i][j][2][0] in stuff or target in stuff):
                OT_OW[target].add(pattern2[i][j][0][0])
                OW_OT[pattern2[i][j][0][0]].add(target)
            
                pattern2_OT_OW[target].add(pattern2[i][j][0][0])
                pattern2_OW_OT[pattern2[i][j][0][0]].add(target)
                

# dont filter here
# it has 2 OW pretty, good we use only pretty good combination
pattern3_OW_OT = defaultdict(set)       
p3_keys = pattern3.keys()
for i in p3_keys:
    if( pattern3[i] != []):
        for j in range(len(pattern3[i])):    
            target = pattern3[i][j][0][0] + " " + pattern3[i][j][1][0]
            OW[i].add(target)

            OW_OT[target].add('NO Opinion Target found')
            
            pattern3_OW_OT[target].add('NO Opinion Target found')
     
            
            
# we use near much, near, much word combinations in OW here
pattern4_OT_OW = defaultdict(set)         
pattern4_OW_OT = defaultdict(set)  
p4_keys = pattern4.keys()
for i in p4_keys:
    if( pattern4[i] != []):
        for j in range(len(pattern4[i])):    
            word = pattern4[i][j][0][0] + " " + pattern4[i][j][1][0] 
            OT[i].add(pattern4[i][j][2][0])
            OW[i].add(word)
            if(pattern4[i][j][2][0] in stuff):
                OT_OW[pattern4[i][j][2][0]].add(word)
                
                OW_OT[word].add(pattern4[i][j][2][0])
                
                pattern4_OT_OW[pattern4[i][j][2][0]].add(word)
                
                pattern4_OW_OT[word].add(pattern4[i][j][2][0])
               
            
# dont filter here            
pattern5_OW_OT = defaultdict(set)           
p5_keys = pattern5.keys()
for i in p5_keys:
    if( pattern5[i] != []):
        for j in range(len(pattern5[i])):    
            target = pattern5[i][j][0][0] + ' ' + pattern5[i][j][1][0]
            OW[i].add(target)
            OW_OT[target].add('No Opinion Target found')
            
            pattern5_OW_OT[target].add('No Opinion Target found')
            

# dont filter here
pattern6_OW_OT = defaultdict(set)
p6_keys = pattern6.keys()
for i in p6_keys:
    if( pattern6[i] != []):
        for j in range(len(pattern6[i])):    
            target = pattern6[i][j][0][0] + " " + pattern6[i][j][2][0]
            OW[i].add(target)
            OW_OT[target].add('NO Opinion Target found')
            
            pattern6_OW_OT[target].add('NO Opinion Target found')
                

pattern7_OW_OT = defaultdict(set)
pattern7_OT_OW = defaultdict(set)
p7_keys = pattern7.keys()
for i in p7_keys:
    if( pattern7[i] != []):
        for j in range(len(pattern7[i])):    
            target = pattern7[i][j][1][0]
            OT[i].add(target)
            OW[i].add(pattern7[i][j][0][0])
            
            if(target in stuff):
                OW_OT[pattern7[i][j][0][0]].add(target)
                OT_OW[target].add(pattern7[i][j][0][0])
                
                pattern7_OW_OT[pattern7[i][j][0][0]].add(target)
                pattern7_OT_OW[target].add(pattern7[i][j][0][0])


# dont filter here
pattern_8_OW_OT = defaultdict(set)
p8_keys = pattern8.keys()
for i in p8_keys:
    if( pattern8[i] != []):
        for j in range(len(pattern8[i])):    
            target = pattern8[i][j][1][0]
            
            OW[i].add(target)
            OW_OT[target].add('No Opinion Target found')
            pattern_8_OW_OT[target].add('No Opinion Target found')
   
################################################################################            
# Step 8: Finding similar Words of above Opinion Words.
            
OW_OT_key = OW_OT.keys()
OW_list = list(OW_OT_key)

from nltk.corpus import wordnet as wn
from itertools import chain

stuff_OW = defaultdict(set)
OW2 = set()

synsets_set = defaultdict(set)
hyponyms_set = defaultdict(set)
for z in range(len(OW_list)): 
    input_word = OW_list[z] 
    OW1= set()    
    for i,j in enumerate(wn.synsets(input_word)):
    #    print ('Meaning', i, 'NLTK ID: ', j.name())
        
        hypernyms = ', '.join(list(chain(*[l.lemma_names() for l in j.hypernyms()])))
    #    print ('Hypernyms:', hypernyms)
        synsets_set[i].add(hypernyms)
        
        hyponyms = ', '.join(list(chain(*[l.lemma_names() for l in j.hyponyms()])))
    #    print ('Hyponyms:', hyponyms)
        hyponyms_set[i].add(hyponyms)
    #    print() 

        ho = [hypernyms]
        for h in range(len(ho)):
            temp_list = ho[h].split(', ')
            if(temp_list != ['']):
                for l in range(len(temp_list)):
                    temp_word = ' '.join(temp_list[l].split('_'))
                    OW2.add(temp_word)
                    OW1.add(temp_word)
        
        hy = [hypernyms]
        for h in range(len(hy)):
            temp_list = hy[h].split(', ')
            if(temp_list != ['']):
                for l in range(len(temp_list)):
                    temp_word = ' '.join(temp_list[l].split('_'))
                    OW2.add(temp_word)
                    OW1.add(temp_word)
    OW1 = list(OW1)
    for i in range(len(OW1)):
        stuff_OW[OW_list[z]].add(OW1[i])


hyponyms_set_keys = hyponyms_set.keys()
synsets_set_keys = synsets_set.keys()
    
for z in range(len(OW_list)):    
    for k in hyponyms_set_keys:
        hy = hyponyms_set[k]
        hy = list(hy)
        for h in range(len(hy)):
            temp_list = hy[h].split(', ')
            if(temp_list != ['']):
                for l in range(len(temp_list)):
                    temp_word = ' '.join(temp_list[l].split('_'))
                    OW2.add(temp_word)
                    
for z in range(len(OW_list)):    
    for k in synsets_set_keys:
        hy = synsets_set[k]
        hy = list(hy)
        for h in range(len(hy)):
            temp_list = hy[h].split(', ')
            if(temp_list != ['']):
                for l in range(len(temp_list)):
                    temp_word = ' '.join(temp_list[l].split('_'))
                    OW2.add(temp_word)
                    
OW_concept = []
OW2 = list(OW2)
for i in range(len(OW2)):
    if(OW2[i] != ''):
        OW_concept.append(OW2[i])
        
for i in range(len(OW_list)):
    OW_concept.append(OW_list[i])
           
# Here OW_concept creates a list of all similar Opinion words of Opinion words generated from pattern
#####################################################################################
# Step 9: Finding Opinion Weight of above similar Opinion Words ( from testimonial.sentiment.polarity )

OW_OT = OW_concept                   
OW_OT_key = OW_OT
OT_OW_key = OT_OW.keys()

OW_in_corpus = defaultdict(set)
for i in corpus.keys():
    for j in range(len(corpus[i])):
        word = corpus[i][j].words
        for k in range(len(word)):
            if( word[k] in OW_OT_key):
              OW_in_corpus[i].add(word[k])
                           
for i in OW_in_corpus.keys():
    OW_in_corpus[i] = list(OW_in_corpus[i])            
                          
testimonial_sentiment = defaultdict(list)
testimonial_sentiment_polarity = defaultdict(list)

for i in OW_in_corpus.keys():
    for j in range(len(OW_in_corpus[i])):
        testimonial = TextBlob(OW_in_corpus[i][j])
        testimonial_sentiment[OW_in_corpus[i][j]].append(testimonial.sentiment)
        testimonial_sentiment_polarity[OW_in_corpus[i][j]].append(testimonial.sentiment.polarity)

OW_in_corpus_list = defaultdict(list)
OW_in_corpus_value = defaultdict(list)
for i in OW_in_corpus.keys():
    for j in range(len(OW_in_corpus[i])):
        word = [OW_in_corpus[i][j]]
#        for k in range(len(word)):
        if(word[0] in testimonial_sentiment_polarity.keys()):
            polarity = testimonial_sentiment_polarity[word[0]]
            OW_in_corpus_value[i].append(polarity[0])
            OW_in_corpus_list[i].append(word[0])

dictionary1 = dict()
dictionary2 = dict()
key_value_pair = defaultdict(list)
for i in range(len(OW_in_corpus_list)):
    for j in range(len(OW_in_corpus_list[i])):
#        dictionary1[OW_in_corpus_value[i][j]] = OW_in_corpus_list[i][j]
        dictionary2[ OW_in_corpus_list[i][j]] = OW_in_corpus_value[i][j]  
        
################################################################################################
# Step 10: Creating Words and its score in tuple_word_score                

score = defaultdict(list)
words = OW_in_corpus_list
OW_in_corpus_value_key = OW_in_corpus_value.keys()
for i in OW_in_corpus_value_key:
    for j in range(len(OW_in_corpus_value[i])):
        score[i].append(OW_in_corpus_value[i][j])
                
tuple_word_score = defaultdict(list)
for i in OW_in_corpus_value_key:
    for j in range(len(OW_in_corpus_value[i])):
        tuple_word_score[i].append((words[i][j], score[i][j]))
        
Opinion_Words = tuple_word_score

list_Opinion_Words = []
for i in range(len(Opinion_Words)):
    list_Opinion_Words.append(Opinion_Words[i])

################################################################################################
# Step 11: Finding Maximum scored words.                

max_abs_score = defaultdict(list)
for i in range(len(tuple_word_score)):
    maxx = 0
    for j in range(len(tuple_word_score[i])):
        temp = abs(tuple_word_score[i][j][1])
        if(temp > maxx):
            maxx = abs(tuple_word_score[i][j][1])
            maxx_word = tuple_word_score[i][j][0]
    max_abs_score[i].append((maxx_word, maxx))
##################################################

maxx_OW_value = defaultdict(set)
for i in range(len(max_abs_score)):
    maxx_OW_value[max_abs_score[i][0][0]].add(max_abs_score[i][0][1])
    
New_OW = maxx_OW_value.keys()

################################################
# step11: Collecting important opinion words.

list_max_abs_score = []
for i in range(len(max_abs_score)):
    list_max_abs_score.append(max_abs_score[i])
    
##########Searching######################################
# step12: Searching it again

OW_in_corpus2 = defaultdict(set)
for i in range(len(corpus)):
    for j in range(len(corpus[i])):
        word = corpus[i][j].words
        for k in range(len(word)):
            if( word[k] in New_OW):
              OW_in_corpus2[i].add(word[k])
              


             
for i in OW_in_corpus2.keys():
    OW_in_corpus2[i] = list(OW_in_corpus2[i])            
              

             
testimonial_sentiment = defaultdict(list)
testimonial_sentiment_polarity = defaultdict(list)

for i in OW_in_corpus2.keys():
    for j in range(len(OW_in_corpus2[i])):
        testimonial = TextBlob(OW_in_corpus2[i][j])
        testimonial_sentiment[OW_in_corpus2[i][j]].append(testimonial.sentiment)
        testimonial_sentiment_polarity[OW_in_corpus2[i][j]].append(testimonial.sentiment.polarity)

OW_in_corpus_list = defaultdict(list)
OW_in_corpus_value = defaultdict(list)
for i in OW_in_corpus2.keys():
    for j in range(len(OW_in_corpus2[i])):
        word = [OW_in_corpus2[i][j]]
#        for k in range(len(word)):
        if(word[0] in testimonial_sentiment_polarity.keys()):
            polarity = testimonial_sentiment_polarity[word[0]]
            OW_in_corpus_value[i].append(polarity[0])
            OW_in_corpus_list[i].append(word[0])


dictionary1 = dict()
dictionary2 = dict()
key_value_pair = defaultdict(list)
for i in range(len(OW_in_corpus_list)):
    for j in range(len(OW_in_corpus_list[i])):
#        dictionary1[OW_in_corpus_value[i][j]] = OW_in_corpus_list[i][j]
        dictionary2[ OW_in_corpus_list[i][j]] = OW_in_corpus_value[i][j]  

# OW_in_corpus_value indicates the polarity score of opinion word
###############################################################################
import numpy as np
              
score = defaultdict(list)
words = OW_in_corpus_list
OW_in_corpus_value_key = OW_in_corpus_value.keys()
for i in OW_in_corpus_value_key:
    for j in range(len(OW_in_corpus_value[i])):
        score[i].append(OW_in_corpus_value[i][j])
        
        
tuple_word_score = defaultdict(list)
for i in OW_in_corpus_value_key:
    for j in range(len(OW_in_corpus_value[i])):
        tuple_word_score[i].append((words[i][j], score[i][j]))


##########################################################################


final_score = defaultdict(list)
for i in range(len(tuple_word_score)):
    final_score[i].append(0)
    
for i in range(len(tuple_word_score)):
    for j in range(len(tuple_word_score[i])):
        final_score[i].append(tuple_word_score[i][j][1])
        
final_score = defaultdict(list)
for i in range(len(tuple_word_score)):
    final_score[i].append(0)
    
for i in range(len(tuple_word_score)):
    for j in range(len(tuple_word_score[i])):
        final_score[i].append(tuple_word_score[i][j][1])
        
############################################################################################
#Step 13: Creating Average
average_score = defaultdict(list)
for i in range(len(final_score)):
    average_score[i].append(np.mean(final_score[i]))


for i in range(len(average_score)):
    if(np.isnan(average_score[i]) == True):
        average_score[i] = [0]

list_average_score = []
for i in range(len(average_score)):
    list_average_score.append(average_score[i])
############################################################################################
    
import numpy as np
  
X_train = []   
for i in average_score.keys():
    temp = set()
    for j in range(len(OW_in_corpus[i])):
        temp.add(OW_in_corpus[i][j])
    X_train.append(temp)
    
l = set()
for i in range(len(X_train)):
    X_train[i] = list (X_train[i])
    for j in range(len(X_train[i])):
        l.add(X_train[i][j])

l =list(l)

import pandas as pd
da = pd.DataFrame(columns = l, index = OW_in_corpus.keys(), data = 0)

k = list(OW_in_corpus.keys())

for m in range(len(k)):
    for j in range(len(l)):
        if(l[j] in OW_in_corpus[k[m]]):
            da.iloc[m][l[j]] = 1

X_train = da.iloc[:, 0:]
X_train = np.array(X_train)

y_train = []
for i in average_score.keys():
    y_train.append(average_score[i][0])    

r = (max(y_train) - min(y_train)) / 4
m = min(y_train)

y_train = np.array(y_train)

y = []
for i in range(len(y_train)):
    y.append(y_train[i])

X = X_train

for i in range(len(y)):
    if(y[i] <= m):
        y[i] = 'Negative'
    elif(y[i] <= m+r and y[i] > m):
        y[i] = 'Negative'
    elif(y[i] <= m+r+r and y[i] > m+r):
        y[i] = 'Negative'
    elif(y[i] <= m+r+r+r and y[i] > m+r+r):
        y[i] = 'Positive'
    elif(y[i] <= m+r+r+r+r and y[i] > m+r+r+r):
        y[i] = 'Positive'
        
y_set = defaultdict(list)


key = average_score.keys()
key = list(key)

for i in range(len(key)):
    y_set[key[i]].append(y[i])


da['calculated rating'] = [y_set[i] for i in y_set.keys()]


rev_rate = []
rate = dataset['rating']
for i in y_set.keys():
    rev_rate.append(rate[i])
    
for i in range(len(rev_rate)):
    if(rev_rate[i] == 1.0):
        rev_rate[i] = 'Negative'
    elif(rev_rate[i] == 2.0):
        rev_rate[i] = 'Negative'
    elif(rev_rate[i] == 3.0):
        rev_rate[i] = 'Negative'
    elif(rev_rate[i] == 4.0):
        rev_rate[i] = 'Positive'
    elif(rev_rate[i] == 5.0):
        rev_rate[i] = 'Positive'


list_tuple_word_score = []
for i in range(len(tuple_word_score)):
    list_tuple_word_score.append(tuple_word_score[i])

comparision_dataframe = pd.DataFrame()

comparision_dataframe['Original Review'] = dataset['reviewText']
comparision_dataframe['Opinion Words'] = list_Opinion_Words
comparision_dataframe['Maximum scored Opinion Words'] = list_max_abs_score
comparision_dataframe['Final Opnion Words'] = list_tuple_word_score
comparision_dataframe['Average'] = list_average_score
comparision_dataframe['Reviewer Rating'] = rev_rate
comparision_dataframe['Calculated rating'] = y


save = comparision_dataframe.to_csv(sep=',')
text_file = open("train_file.csv", "w")
text_file.write(save)
text_file.close()

#######################################################################
# Step 14:  Making the Confusion Matrix

y = comparision_dataframe['Calculated rating']

from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
cm = confusion_matrix(rev_rate, y)

accuracy = 0
for i in range(len(cm)):
    for j in range(len(cm)):
        if(i == j):
            accuracy = accuracy + cm[i][j]
accuracy = accuracy/ len(rev_rate) * 100


TP = np.diag(cm)
FP = np.sum(cm, axis=0) - TP
FN = np.sum(cm, axis=1) - TP

num_classes = len(cm)
TN = []
for i in range(num_classes):
    temp = np.delete(cm, i, 0)    # delete ith row
    temp = np.delete(temp, i, 1)  # delete ith column
    TN.append(sum(sum(temp)))
  
    
#Let's make a sanity check: for each class, the sum of TP, FP, FN, and TN 
# must be equal to the size of our test set (here 10,000): 
#let's confirm that this is indeed the case:
    
l = len(rev_rate)
for i in range(num_classes):
    print(TP[i] + FP[i] + FN[i] + TN[i] == l)
    
   
# link : http://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/

precision = TP/(TP+FP)
recall = TP/(TP+FN)
FScore = 2*(recall * precision) / (recall + precision)

DataFrame = pd.DataFrame()
DataFrame['precision'] = precision
DataFrame['recall'] = recall
DataFrame['FScore'] = FScore

###################################################################################################
#################################################################################################3
# Step 15: Testing

New_sentence = "this phone is gifted to me by my friend.oh my god!!. i am not much impreseed by the sound quality. It isn't the one i expected. i am disappointed by its short term life."
New_rating = 'Negative'
New_sentence = ' This is a wonderful screen touch!! can it get better than this? i am so much excited!!'
New_rating = 'Positive'


#step 1: pre processing

# isn't = is not
New_sentence = decontracted(New_sentence)

# removed Unwanted symbols, lower case
comment_dict2 = defaultdict(list)
for i in range(1):
    sentence = re.sub('[^a-zA-Z.]',' ', New_sentence)
    sentence = sentence.lower()
    sentence = sentence.split('.')
    for k in range(len(sentence)):
        review = sentence[k].split()
        review = [word for word in review if not word in set(stopwords.words('english'))]
        sentence[k] =  ' '.join(review)
        comment_dict2[i].append(sentence[k])

#delete unwanted '' words
for j in range(len(comment_dict2)):
    comment_dict2[j] = [comment_dict2[j][i] for i in range(len(comment_dict2[j])) if comment_dict2[j][i] not in '']

Text = defaultdict(list)
for i in range(len(comment_dict2)):
    Text[i].append(('. '.join(comment_dict2[i][j] for j in range(len(comment_dict2[i])))))

# spelling correction
for i in range(len(Text)):
    b = TextBlob(Text[i][0])
    Text[i] = b.correct()

# creating corpus
corpus2 = defaultdict(set)
for i in range(len(Text)):
    wiki = Text[i]
    corpus2[i] = wiki.sentences
            
corpus_key2 = corpus2.keys()
corpus_list2 = defaultdict(list)

for i in corpus_key2:
    for j in range(len(corpus2[i])):
        word = ' '.join(corpus2[i][j].words)
        corpus_list2[i].append(word)

Text = ""
for i in range(len(corpus_list2)):
    temp = corpus_list2[i]
    for j in temp:
        Text += str(j) + ' '
##########################################################################
# Step2 of testing : Use OW and find its score
        
list_max_abs_score = []
for i in range(len(max_abs_score)):
    list_max_abs_score.append(max_abs_score[i])

t = TextBlob(Text)
OW_in_corpus22 = defaultdict(set)
for i in range(len(corpus2)):
    for j in range(len(corpus2[i])):
        word = t.words
        for k in range(len(word)):
            if( word[k] in New_OW):
              OW_in_corpus22[i].add(word[k])
              
for i in OW_in_corpus22.keys():
    OW_in_corpus22[i] = list(OW_in_corpus22[i])            
              
testimonial_sentiment = defaultdict(list)
testimonial_sentiment_polarity = defaultdict(list)
OW_in_corpus2 = OW_in_corpus22

for i in OW_in_corpus2.keys():
    for j in range(len(OW_in_corpus2[i])):
        testimonial = TextBlob(OW_in_corpus2[i][j])
        testimonial_sentiment[OW_in_corpus2[i][j]].append(testimonial.sentiment)
        testimonial_sentiment_polarity[OW_in_corpus2[i][j]].append(testimonial.sentiment.polarity)

OW_in_corpus_list = defaultdict(list)
OW_in_corpus_value = defaultdict(list)
for i in OW_in_corpus2.keys():
    for j in range(len(OW_in_corpus2[i])):
        word = [OW_in_corpus2[i][j]]
#        for k in range(len(word)):
        if(word[0] in testimonial_sentiment_polarity.keys()):
            polarity = testimonial_sentiment_polarity[word[0]]
            OW_in_corpus_value[i].append(polarity[0])
            OW_in_corpus_list[i].append(word[0])

dictionary1 = dict()
dictionary2 = dict()
key_value_pair = defaultdict(list)
for i in range(len(OW_in_corpus_list)):
    for j in range(len(OW_in_corpus_list[i])):
#        dictionary1[OW_in_corpus_value[i][j]] = OW_in_corpus_list[i][j]
        dictionary2[ OW_in_corpus_list[i][j]] = OW_in_corpus_value[i][j]  

import numpy as np            
score = defaultdict(list)
words = OW_in_corpus_list
OW_in_corpus_value_key = OW_in_corpus_value.keys()
for i in OW_in_corpus_value_key:
    for j in range(len(OW_in_corpus_value[i])):
        score[i].append(OW_in_corpus_value[i][j])
       
tuple_word_score = defaultdict(list)
for i in OW_in_corpus_value_key:
    for j in range(len(OW_in_corpus_value[i])):
        tuple_word_score[i].append((words[i][j], score[i][j]))

final_score = defaultdict(list)
for i in range(len(tuple_word_score)):
    final_score[i].append(0)
    
for i in range(len(tuple_word_score)):
    for j in range(len(tuple_word_score[i])):
        final_score[i].append(tuple_word_score[i][j][1])
        
final_score = defaultdict(list)
for i in range(len(tuple_word_score)):
    final_score[i].append(0)
    
for i in range(len(tuple_word_score)):
    for j in range(len(tuple_word_score[i])):
        final_score[i].append(tuple_word_score[i][j][1])

average_score = defaultdict(list)
for i in range(len(final_score)):
    average_score[i].append(np.mean(final_score[i]))


for i in range(len(average_score)):
    if(np.isnan(average_score[i]) == True):
        average_score[i] = [0]

list_avg = []
for i in range(len(average_score)):
    list_avg.append(average_score[i][0])

list_tuple_word_score = []
for i in range(len(tuple_word_score)):
    list_tuple_word_score.append(tuple_word_score[i])
###############################################################
y_train = []
for i in average_score.keys():
    y_train.append(average_score[i][0])    

y_train = np.array(y_train)

y = []
for i in range(len(y_train)):
    y.append(y_train[i])

X = X_train

for i in range(len(y)):
    if(y[i] <= m):
        y[i] = 'Negative'
    elif(y[i] <= m+r and y[i] > m):
        y[i] = 'Negative'
    elif(y[i] <= m+r+r and y[i] > m+r):
        y[i] = 'Negative'
    elif(y[i] <= m+r+r+r and y[i] > m+r+r):
        y[i] = 'Positive'
    elif(y[i] <= m+r+r+r+r and y[i] > m+r+r+r):
        y[i] = 'Positive'
    
c_dataframe = pd.DataFrame()
c_dataframe['Original Testing Review'] = [New_sentence]
c_dataframe['Processed Testing Review'] = [Text]
c_dataframe['Opinion Words Obtained'] = list_tuple_word_score
c_dataframe['Average of Opinion Words'] = average_score[0]
c_dataframe['Original Rating'] = New_rating
c_dataframe['Predicted Rating'] = y

###################### Validation by Machine Learning ####################################################################
X = X_train
y = comparision_dataframe['Calculated rating']
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#####################################################################

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)

# Fitting DecisionTreeClassifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier( criterion = 'entropy', random_state = 0)

# Fitting Naive Bayes to the Training set
#from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
#from sklearn.naive_bayes import BernoulliNB
#classifier = GaussianNB()
classifier = MultinomialNB() # do no standartize
#classifier = BernoulliNB()

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier( n_estimators = 10, criterion = 'entropy', random_state = 0, n_jobs = -1)

# Fitting  K-Nearest Neighbors K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

# Fitting Linear Support Vector Machine Linear SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel= 'sigmoid', random_state = 0)

# Fitting Linear Support Vector Machine Linear SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel= 'rbf', random_state = 0)


# Fitting Linear Support Vector Machine Linear SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel= 'poly', random_state = 0)

# Fitting Linear Support Vector Machine Linear SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel= 'linear', random_state = 0)

#####################################################################

classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy = 0
for i in range(len(cm)):
    for j in range(len(cm)):
        if(i == j):
            accuracy = accuracy + cm[i][j]
accuracy = accuracy/ len(y_test) * 100


# 10 fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv=10, n_jobs = -1)

accuracy_by_10fold_cv = (sum(accuracies) / 10)
accuracy_by_10fold_cv = accuracies.mean() * 100
std = accuracies.std() * 100
accuracy = accuracy_by_10fold_cv

TP = np.diag(cm)
FP = np.sum(cm, axis=0) - TP
FN = np.sum(cm, axis=1) - TP

num_classes = len(cm)
TN = []
for i in range(num_classes):
    temp = np.delete(cm, i, 0)    # delete ith row
    temp = np.delete(temp, i, 1)  # delete ith column
    TN.append(sum(sum(temp)))
  
    
#Let's make a sanity check: for each class, the sum of TP, FP, FN, and TN 
# must be equal to the size of our test set (here 10,000): 
#let's confirm that this is indeed the case:
    
l = len(y_test)
for i in range(num_classes):
    print(TP[i] + FP[i] + FN[i] + TN[i] == l)
    
   
# link : http://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/

precision = TP/(TP+FP)
recall = TP/(TP+FN)
FScore = 2*(recall * precision) / (recall + precision)

DataFrame = pd.DataFrame()
DataFrame['precision'] = precision
DataFrame['recall'] = recall
DataFrame['FScore'] = FScore

print('\nDataFrame:\n', DataFrame)
print('\n\nConfusion Matrix:\n', cm)
print('\n\nAccuracy: ', accuracy)



print('\n\nOpinionWords:\n', OW_in_corpus)