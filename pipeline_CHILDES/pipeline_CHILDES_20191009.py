###############################################################
################## PIPELINE FOR CHILDES (spell-checked; POS-tagged) ####################
###############################################################

## import package / module
import pickle #storing models
import random #generate pseudo-random numbers for various distributions
from collections import defaultdict #collections: container datatypes; defaultdict: dict subclass that calls a factory function to supply missing values
import logging #flexible framework for emitting log messages from Python programmes
from collections import Counter #creates frequency list efficiently
from operator import itemgetter #recalls the function 'itemgetter'
from sklearn.metrics import classification_report #calculate precision, recall, fcore, & support


## Define functions

# split one training data into training, development, & test set

def simple_train_test(in_text, perc_train):
    train_index = int(len(in_text) * perc_train)

    return(in_text[:train_index],in_text[train_index:])


# de-tach word from tag (for test set)

def tag_strip_sents(tagged_text):
    untagged_sents = [] #holder for untagged sentence

    for sent in tagged_text: #iterate
        ut_sent = [] #holder for untagged sentence; temporary
        for x in sent: #iterate
            ut_sent.append(x[0]) #append 0th item of each pair (=key =word) to ut_sent
        untagged_sents.append(ut_sent) #append ut_sent to untagged_sents

    return(untagged_sents) #return result


# calculate accuracy: hand-tagged list (gold) & machine-tagged text (test)

def simple_accuracy_sent(gold,test):
    correct = 0 #holder for correct count
    nwords = 0 #holder for total words count

    for sent_id, sents in enumerate(gold): #iterate through sentences #enumerate(): add index; here, index = sent_id & item = sents
        for word_id, (gold_word, gold_tag) in enumerate(sents): #iterate through words in each sentence using enumerate() #format = [index, [word, tag]]; index = word_id, word = word, tag = tag
            nwords += 1 #add 1 to nwords
            if gold_tag == test[sent_id][word_id][1]: #if tag correct > add one to correct score
                correct +=1 #add 1 to correct

    return(correct/nwords) #return accuracy


# detach tag from word

def tag_for_calculation(data):
    tag = [] #holder for tag
    for x in data: #iterate through [word, tag] sents
        for word_tag in x: #iterate through [word, tag]
            tag.append(word_tag[1]) #append only tag to holder
    return(tag)





###################### Create training data ######################


## Load raw data (with tag info)
corpus_raw = open("CHILDES_final_20191009.txt").read().split("\n\n\n") #orgn data format: sent \n CONLL by eojeol 
print(len(corpus_raw)) #96044


## pre-processing (1): exclude single-morpheme utterance

text_conll = []

for sent_raw in corpus_raw:
    if sent_raw == "\n":
        continue
    elif sent_raw == "":
        continue
    else:
        lines_raw = sent_raw.split("\n")
        line_fullsent = lines_raw[0]
        if len(line_fullsent) <= 15:
            continue
        else:
            text_conll.append(lines_raw)

print(text_conll[:3])
print(len(text_conll)) #69498


## pre-processing (2): merging info by sent

sents_for_tagging_list = []

for sent in text_conll:
    eojeol_fullinfo_temp = []

    for line in sent:
        if line == "\n":
            continue
        elif line == "":
            continue
        elif "#" in line:
            continue
        else:
            items = line.split("\t")
            eojeol = items[1]
#            print(eojeol)
            eojeol_by_morph = items[2]
            upos = items[3]
            xpos = items[4]
            eojeol_fullinfo = eojeol + "/" + eojeol_by_morph + "/" + xpos + "/" + upos
            if "PUNCT" in eojeol_fullinfo:
                continue
            else:
                eojeol_fullinfo_temp.append(eojeol_fullinfo)
        
    sent_fullinfo = " ".join(eojeol_fullinfo_temp)
    sents_for_tagging_list.append(sent_fullinfo)

print(sents_for_tagging_list[-1]) #['기호는/기호+는/NNP+JX/NOUN 날마다/날+마다/NNG+JX/ADV 책을/책+을/NNG+JKO/NOUN 열심히/열심히/MAG/ADV 읽었어요/읽+었+어요/VV+EP+EF/VERB']
print(len(sents_for_tagging_list))


sents_for_tagging_list_sanity_chk = []

for sent in sents_for_tagging_list:
    if sent == "":
        continue
    elif sent == " ":
        continue
    else:
        sents_for_tagging_list_sanity_chk.append(sent)

print(sents_for_tagging_list_sanity_chk[-1])
print(len(sents_for_tagging_list_sanity_chk))



## pre-processing (3): merging info by sent

sent_by_eojeol_xpos = []
sent_by_eojeol_xpos_upos = []

for sent in sents_for_tagging_list_sanity_chk:
    sent_by_eojeol_xpos_temp = []
    sent_by_eojeol_xpos_upos_temp = []
    sent_eojeol = sent.split(" ")
    
    for indiv_eojeol in sent_eojeol:
        if indiv_eojeol == "\n":
            continue
        elif indiv_eojeol == "":
            continue
        else:
            items = indiv_eojeol.split("/")
            eojeol_full = items[0]
            eojeol_morph_combi = items[1]
            upos = items[3]
            xpos = items[2]
            eojeol_morph_combi_xpos = eojeol_morph_combi + "/" + xpos
            sent_by_eojeol_xpos_temp.append(eojeol_morph_combi_xpos)
            eojeol_morph_combi_xpos_upos = eojeol_morph_combi + "/" + xpos + "/" + upos
            sent_by_eojeol_xpos_upos_temp.append(eojeol_morph_combi_xpos_upos)

    sent_by_eojeol_xpos_complete = " ".join(sent_by_eojeol_xpos_temp)
    sent_by_eojeol_xpos.append(sent_by_eojeol_xpos_complete)
    sent_by_eojeol_xpos_upos_complete = " ".join(sent_by_eojeol_xpos_upos_temp)
    sent_by_eojeol_xpos_upos.append(sent_by_eojeol_xpos_upos_complete)

print(sent_by_eojeol_xpos[:3]) #['둘/NR 중+에/NNB+JKB 하나+만/NR+JX 하+아/VV+EF 예주/NNP', '어/MAG 하나+만/NR+JX', '응/IC 하나+만/NR+JX']
print(len(sent_by_eojeol_xpos))

print(sent_by_eojeol_xpos_upos[:3]) #['둘/NR/NOUN 중+에/NNB+JKB/ADP 하나+만/NR+JX/NUM 하+아/VV+EF/VERB 예주/NNP/NOUN', '어/MAG/ADV 하나+만/NR+JX/NUM', '응/IC/NOUN 하나+만/NR+JX/NUM']
print(len(sent_by_eojeol_xpos_upos))


## pre-processing (4): create (morpheme_index, xpos) set

pretty_corpus_for_xpos_tagging = []

for line in sent_by_eojeol_xpos:
    sentence = []
    morpheme_xpos_restructure = []
    eojeol_xpos = line.split(" ") #[중+에/NNB+JKB]

    for x in eojeol_xpos:
        indiv_item = x.split("/") #[중+에, NNB+JKB]
        morpheme = indiv_item[0] #[중+에]
        xpos_morpheme = indiv_item[1] #[NNB+JKB]
        morpheme_xpos_indiv_combi = []
        morpheme_indiv_temp = morpheme.split("+") #[중+에]
        xpos_morpheme_temp = xpos_morpheme.split("+") #[NNB+JKB]
        
        for x_temp, y_temp in zip (morpheme_indiv_temp, xpos_morpheme_temp):
            morpheme_xpos_indiv_combi_temp = x_temp + "/" + y_temp #[중/NNB, 에/JKB]
            morpheme_xpos_indiv_combi.append(morpheme_xpos_indiv_combi_temp)
        
        #print(morpheme_xpos_indiv_combi)

        morpheme_xpos_restructure_temp = "+".join(morpheme_xpos_indiv_combi) #[중/NNB+에/JKB]
        morpheme_xpos_restructure.append(morpheme_xpos_restructure_temp)

    #print(morpheme_xpos_restructure) #['둘/NR', '중/NNB+에/JKB', '하나/NR+만/JX', '하/VV+아/EF', '예주/NNP']

    for x in morpheme_xpos_restructure:
        word = x.split("+") #[중/NNB, 에/JKB]
        for idx, y in enumerate(word):
            if len(y) < 1: #kick out any unnecessary stuff
                continue
            #print(y) #sanity check
            morph = y.split("/")[0] #assign morph as the first data point (i.e., key)
            if morph == '': #kick out any unnecessary stuff
                continue
            else: #attach index number to morph
                morph = morph + "_" + str(idx)
            # bypass any error (b/c error chars not critical for our purpose)
            try:
                tag = y.split("/")[1] #assign tag as the second data point (i.e., value)
            except IndexError:
                continue
            sentence.append((morph,tag)) #key-value pairing #((중_0,NNB), (에_1,JKB))

    pretty_corpus_for_xpos_tagging.append(sentence)

print(pretty_corpus_for_xpos_tagging[:3]) #works fine
print(len(pretty_corpus_for_xpos_tagging))


## pre-processing (5): create (morpheme_index/xpos_index, upos) set

pretty_corpus_for_upos_tagging = []

for line in sent_by_eojeol_xpos_upos:
    sentence = []
    eojeol_upos = line.split(" ") #[하+아/VV+EF/VERB, ...]

    for items in eojeol_upos:
        item = items.split("/") #[하+아, VV+EF, VERB]
        ind_morph = item[0].split("+") #[하, 아]
        ind_xpos = item[1].split("+") #[VV, EF]
        upos = item[2] #[VERB]

        morpheme_list = []
        xpos_list = []

        for idx, morph in enumerate(ind_morph):
            morph = morph + "_" + str(idx)
            morpheme_list.append(morph)
    
        morpheme_idx_combi = "+".join(morpheme_list)

        for idx, xpos in enumerate(ind_xpos):
            xpos = xpos + "_" + str(idx)
            xpos_list.append(xpos)
    
        xpos_idx_combi = "+".join(xpos_list)

        morpheme_xpos = morpheme_idx_combi + "/" + xpos_idx_combi # [하_0+아_1/VV_0+EF_1]
    
        sentence.append((morpheme_xpos, upos))
        
    pretty_corpus_for_upos_tagging.append(sentence)

print(pretty_corpus_for_upos_tagging[:3]) # [[('하_0+아_1/VV_0+EF_1', 'VERB'), ...], ...]
print(len(pretty_corpus_for_upos_tagging))


# Output file (thru pickle)

with open('pretty_corpus_CHILDES_xpos_tagging_20191009.pkl', 'wb') as outfile:
    pickle.dump(pretty_corpus_for_xpos_tagging,outfile)

with open('pretty_corpus_CHILDES_upos_tagging_20191009.pkl', 'wb') as outfile:
    pickle.dump(pretty_corpus_for_upos_tagging,outfile)





###################### Perceptron tagger: XPOS tagging ######################


## Machine-learning part
# average perceptron (by Matthew Honnibal) #https://explosion.ai/blog/part-of-speech-pos-tagger-in-python

class AveragedPerceptron(object):

    def __init__(self):
        self.weights = {} #each feature gets its own weight vector; weights = dict-of-dicts
        self.classes = set()
        self._totals = defaultdict(int) #the accumulated values (for the averaging); keyed by feature/class tuples
        self._tstamps = defaultdict(int) #the last time the feature was changed (for the averaging); keyed by feature/class tuples; tstamps =  timestamps
        self.i = 0 #number of instances seen

    def predict(self, features):
        '''Dot-product the features and current weights and return the best label.'''
        scores = defaultdict(float) #
        for feat, value in features.items(): #iterate through features
            if feat not in self.weights or value == 0: #feature not found or 0 value > continue to next iteration
                continue
            weights = self.weights[feat] #feature in self.weights as weights
            for label, weight in weights.items(): #iterate through weights
                scores[label] += value * weight #calcualte score of each label

        return max(self.classes, key=lambda label: (scores[label], label)) #do secondary alphabetic sort for stability

    def update(self, truth, guess, features):
        '''Update the feature weights.'''
        def upd_feat(c, f, w, v): #c=class, f=feature, w=weight, v=vlaue
            param = (f, c) #feature-class pair
            self._totals[param] += (self.i - self._tstamps[param]) * w #calcualte weight for each pair
            self._tstamps[param] = self.i #???
            self.weights[f][c] = w + v #???

        self.i += 1 #increase self.i
        if truth == guess: #two weights equal > fine
            return None
        for f in features: #iterate through features
            weights = self.weights.setdefault(f, {}) #default weight
            upd_feat(truth, f, weights.get(truth, 0.0), 1.0) #update weight for feature: truth+1
            upd_feat(guess, f, weights.get(guess, 0.0), -1.0) #update weight for feature: guess-1

    def average_weights(self):
        '''Average weights from all iterations.'''
        for feat, weights in self.weights.items(): #iterate through weights.items
            new_feat_weights = {} #holder (dictionary) for average weight of feature
            for clas, weight in weights.items(): #iterate through weights.items
                param = (feat, clas) #feature-class pair
                total = self._totals[param] #total weight of each pair
                total += (self.i - self._tstamps[param]) * weight #calcualte weight for each pair
                averaged = round(total / self.i, 3) #calcualte average weight
                if averaged:
                    new_feat_weights[clas] = averaged #update weight
            self.weights[feat] = new_feat_weights #assign new weight to feature

    def save(self, path):
        '''Save the pickled model weights.'''
        with open(path, 'wb') as fout: #convention for saving pickle #wb = write byte
            return pickle.dump(dict(self.weights), fout)

    def load(self, path):
        '''Load the pickled model weights.'''
        self.weights = load(path) #convention for loading pickle


## Tagger part
# Greedy Averaged Perceptron tagger by Matthew Honnibal
# https://explosion.ai/blog/part-of-speech-pos-tagger-in-python

class AveragedPerceptron(object):

    def __init__(self):
        self.weights = {} #each feature gets its own weight vector; weights = dict-of-dicts
        self.classes = set()
        self._totals = defaultdict(int) #the accumulated values (for the averaging); keyed by feature/class tuples
        self._tstamps = defaultdict(int) #the last time the feature was changed (for the averaging); keyed by feature/class tuples; tstamps =  timestamps
        self.i = 0 #number of instances seen

    def predict(self, features):
        '''Dot-product the features and current weights and return the best label.'''
        scores = defaultdict(float) #
        for feat, value in features.items(): #iterate through features
            if feat not in self.weights or value == 0: #feature not found or 0 value > continue to next iteration
                continue
            weights = self.weights[feat] #feature in self.weights as weights
            for label, weight in weights.items(): #iterate through weights
                scores[label] += value * weight #calcualte score of each label

        return max(self.classes, key=lambda label: (scores[label], label)) #do secondary alphabetic sort for stability

    def update(self, truth, guess, features):
        '''Update the feature weights.'''
        def upd_feat(c, f, w, v): #c=class, f=feature, w=weight, v=vlaue
            param = (f, c) #feature-class pair
            self._totals[param] += (self.i - self._tstamps[param]) * w #calcualte weight for each pair
            self._tstamps[param] = self.i #???
            self.weights[f][c] = w + v #???

        self.i += 1 #increase self.i
        if truth == guess: #two weights equal > fine
            return None
        for f in features: #iterate through features
            weights = self.weights.setdefault(f, {}) #default weight
            upd_feat(truth, f, weights.get(truth, 0.0), 1.0) #update weight for feature: truth+1
            upd_feat(guess, f, weights.get(guess, 0.0), -1.0) #update weight for feature: guess-1

    def average_weights(self):
        '''Average weights from all iterations.'''
        for feat, weights in self.weights.items(): #iterate through weights.items
            new_feat_weights = {} #holder (dictionary) for average weight of feature
            for clas, weight in weights.items(): #iterate through weights.items
                param = (feat, clas) #feature-class pair
                total = self._totals[param] #total weight of each pair
                total += (self.i - self._tstamps[param]) * weight #calcualte weight for each pair
                averaged = round(total / self.i, 3) #calcualte average weight
                if averaged:
                    new_feat_weights[clas] = averaged #update weight
            self.weights[feat] = new_feat_weights #assign new weight to feature

    def save(self, path):
        '''Save the pickled model weights.'''
        with open(path, 'wb') as fout: #convention for saving pickle #wb = write byte
            return pickle.dump(dict(self.weights), fout)

    def load(self, path):
        '''Load the pickled model weights.'''
        self.weights = load(path) #convention for loading pickle


## Tagger part
# Greedy Averaged Perceptron tagger by Matthew Honnibal
# https://explosion.ai/blog/part-of-speech-pos-tagger-in-python

class PerceptronTagger_simple():

    START = ['-START-', '-START2-']
    END = ['-END-', '-END2-']

    def __init__(self, load=True):
        '''
        :param load: Load the pickled model upon instantiation.
        '''
        self.model = AveragedPerceptron() #load AveragePerceptron class
        self.tagdict = {} #holder (dictionary)
        self.classes = set()
        if load:
            AP_MODEL_LOC = PICKLE #specify location of model
            self.load(AP_MODEL_LOC) #load model

    def tag(self, tokens):
        '''
        Tag tokenized sentences.
        :params tokens: list of word
        :type tokens: list(str)
        '''
        prev, prev2 = self.START #set prev & prev2
        post, post2 = self.END #set post & post2
        output = [] #holder for output

        context = self.START + [self.normalize(w) for w in tokens] + self.END #set context using envorinment of start & end and normalised weight for each word token
        for i, word in enumerate(tokens): #iterate through tokens #enumerate(): add index; here, index = i & item = word
            tag = self.tagdict.get(word) #tag as tag of word from tagdict
            if not tag: #not found in tagdict > find/predict the best tag based on probability
                features = self._get_features(i, word, context, prev, prev2, post, post2)
                tag = self.model.predict(features)
            output.append((word, tag)) #append predicted word-tag pair
            prev2 = prev #sequence +1
            prev = tag #sequence +1

        return output

    def train(self, sentences, save_loc=None, nr_iter=10):
        '''Train a model from sentences, and save it at ``save_loc``.
        ``nr_iter`` controls the number of Perceptron training iterations.
        :param sentences: A list or iterator of sentences, where each sentence
            is a list of (words, tags) tuples.
        :param save_loc: If not ``None``, saves a pickled model in this location.
        :param nr_iter: Number of training iterations.
        '''
        self._sentences = list()  #to be populated by self._make_tagdict
        self._make_tagdict(sentences) #create tag dictionary using sentences
        self.model.classes = self.classes
        for iter_ in range(nr_iter): #iterate through as many as number of iteration
            c = 0 #starting point for class
            n = 0 #starting point for number (of iteration)
            for sentence in self._sentences: #iterate through self._sentences
                words, tags = zip(*sentence) #???
                prev, prev2 = self.START #set prev & prev2
                post, post2 = self.END
                context = self.START + [self.normalize(w) for w in words] + self.END #set context using envorinment of start & end and normalised weight for each word token
                for i, word in enumerate(words): #iterate through tokens #enumerate(): add index; here, index = i & item = word
                    guess = self.tagdict.get(word) #guess as tag of word from tagdict
                    if not guess: #not found in guess > find/predict the best tag based on probability > update info
                        feats = self._get_features(i, word, context, prev, prev2, post, post2)
                        guess = self.model.predict(feats)
                        self.model.update(tags[i], guess, feats)
                    prev2 = prev #sequence +1
                    prev = guess #sequence +1
                    c += guess == tags[i] #???
                    n += 1 #increase number of n(iteration)
            random.shuffle(self._sentences) #???
            logging.info("Iter {0}: {1}/{2}={3}".format(iter_, c, n, _pc(c, n))) #???

        self._sentences = None #delete training sentences (We don't need the training sentences anymore, and we don't want to waste space on them when we pickle the trained tagger)

        self.model.average_weights() #calculate average weight for feature in model

        if save_loc is not None: #parameter changed
            with open(save_loc, 'wb') as fout: #convention for saving pickle #wb = write byte
                pickle.dump((self.model.weights, self.tagdict, self.classes), fout, 2)
                #cf) changed protocol from -1 to 2 to make pickling Python 2 compatible

    def load(self, loc):
        '''
        :param loc: Load a pickled model at location.
        :type loc: str
        '''
        self.model.weights, self.tagdict, self.classes = pickle.load(open(loc,"rb"))
        self.model.classes = self.classes


    def normalize(self, word):
        '''
        Normalisation for pre-processing.
        - All words are lower cased
        - Groups of digits of length 4 are represented as !YEAR;
        - Other digits are represented as !DIGITS
        :rtype: str
        '''
        if '-' in word and word[0] != '-':
            return '!HYPHEN'
        elif word.isdigit() and len(word) == 4:
            return '!YEAR'
        elif word[0].isdigit():
            return '!DIGITS'
        else:
            return word.lower()

    def _get_features(self, i, word, context, prev, prev2, post, post2): #word: current morph_idx pair, context: word possible to look forewards or backwards (so movable)
        '''Map tokens into a feature representation, implemented as a
        {hashable: int} dict. If the features change, a new model must be
        trained.
        '''
        def add(name, *args): #function for adding feature-explanation-occurrence info
            features[' '.join((name,) + tuple(args))] += 1

        i += len(self.START)
        features = defaultdict(int)
        add('bias') #act like a prior
        add('i morph', word.split("_")[0]) #current morph
        add('i morph + position', context[i]) #current morph+position set
        add('i-1 word', context[i-1]) #previous word
        add('i-2 word', context[i-2]) #word before previous word
        add('i+1 word', context[i+1]) #next word
        add('i+2 word', context[i+2]) #word next to next word
        add('i-1 tag', prev) #tag of previous word
        add('i-2 tag', prev2) #tag of word before previous word
        add('i+1 tag', post) #tag of next word
        add('i+2 tag', post2) #tag of word next to next word
        add('i-1 tag + i-2 tag', prev, prev2)  #tag sequence of previous two words
        add('i+1 tag + i+2 tag', post, post2)  #tag sequence of next two words
        add('i-1 tag + i morph', prev, word.split("_")[0]) #bigram: tag of previous word + current morph
        add('i morph + i+1 tag', word.split("_")[0], post) #bigram: current morph + tag of previous word
        add('i-1 word + i-1 tag + i morph', context[i-1], prev, word.split("_")[0]) #trigram: tag of previous word + previous word + current morph
        add('i morph + i+1 morph + i+1 tag', word.split("_")[0], context[i+1], post) #trigram: current morph + next word + tag of next word
        add('i-2 tag + i-1 tag + i morph', prev2, prev, word.split("_")[0]) #trigram: tag of previous two words + current morph
        add('i morph + i+1 tag + i+2 tag', word.split("_")[0], post, post2) #trigram: current morph + tag of previous two words
        return features

    def _make_tagdict(self, sentences):
        '''
        Make a tag dictionary for single-tag words.
        :param sentences: A list of list of (word, tag) tuples.
        '''
        counts = defaultdict(lambda: defaultdict(int)) #
        for sentence in sentences: #iterate through sentences
            self._sentences.append(sentence) #append sentence to self._sentences
            for word, tag in sentence: #iterate through sentence
                counts[word][tag] += 1 #increase number
                self.classes.add(tag) #add tag to classes
        freq_thresh = 20 #cut-off level for frequency
        ambiguity_thresh = 0.985 #cut-off level for determination of ambiguity
        for word, tag_freqs in counts.items(): #iterate through counts.items
            tag, mode = max(tag_freqs.items(), key=lambda item: item[1]) #???
            n = sum(tag_freqs.values()) #n as sum of freq value
            if n >= freq_thresh and (mode / n) >= ambiguity_thresh: #don't add rare words to the tag dictionary & only add unambiguous words
                self.tagdict[word] = tag

def _pc(n, d): #???
    return (n / d) * 100


if __name__ == '__main__':
    #_get_pretrain_model()
    pass


## train the tagger

with open("pretty_corpus_CHILDES_xpos_tagging_20191009.pkl","rb") as infile: #convention for loading pickle #rb = read byte
    whole_data = pickle.load(infile)

training = simple_train_test(whole_data,.9)[0]
test = simple_train_test(whole_data,.9)[1]
#dvlp_test = simple_train_test(whole_data,8.)[1]
#development = simple_train_test(dvlp_test,.5)[0]
#test = simple_train_test(dvlp_test,.5)[1]

#test_sents_dvlp = tag_strip_sents(development)
test_sents_test = tag_strip_sents(test)

print(training[-1]) #works fine #no overlap
print(len(training)) #works fine #62548
#print(dvlp_test[:10]) #works fine
#print(len(dvlp_test)) #works fine
#print(development[:5]) #works fine #no overlap
#print(len(development)) #works fine
print(test[-1]) #works fine #no overlap #[[(word, tag), ...], [(word, tag), ...]]
print(len(test)) #works fine #6950
#print(test_sents_dvlp[:5]) #works fine
#print(len(test_sents_dvlp)) #works fine
print(test_sents_test[-1]) #works fine; [[word, ...], [word, ...]]
print(len(test_sents_test)) #works fine #6950


## Accuracy, precision, recall, & F1 value for xpos tagging

# Define and train tagger

pt_tagger = PerceptronTagger_simple(load=False) #define tagger #load=False: do not lode the default model (i.e., pre-trained model) b/c we are going to train our model and use it
pt_tagger.train(training,save_loc = "CHILDES_xpos_perceptron_iter10_freq20_ambi985_20191009.pkl") #train and save model


#pt_tagger.load("CHILDES_xpos_perceptron_iter10_freq10_ambi985_20190928.pkl")


# Tag sentences (the tagger tags one sentence by default)

#pt_tagger_tagged_dvlp = []
#for x in test_sents_dvlp:
#    pt_tagger_tagged_dvlp.append(pt_tagger.tag(x))

pt_tagger_tagged_test = []
for x in test_sents_test:
    pt_tagger_tagged_test.append(pt_tagger.tag(x))


# Check accuracy
    
#print(simple_accuracy_sent(development,pt_tagger_tagged_dvlp)) 
print(simple_accuracy_sent(test,pt_tagger_tagged_test)) #94.50


# Precision, recall, f1-score

test_tag_only = tag_for_calculation(test) #de-tach tag from word-tag pair
print(test_tag_only[:10]) #works fine
print(len(test_tag_only)) #works fine

pt_tagger_tagged_tag_only = tag_for_calculation(pt_tagger_tagged_test) #de-tach tag from word-tag pair
print(pt_tagger_tagged_tag_only[:10]) #works fine
print(len(pt_tagger_tagged_tag_only)) #works fine

xpos_list =["NNG", "NNP", "NNB", "NR", "NP", "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JC", "JX", "VV", "VX", "VCP", "VCN", "VA", "MM", "EP", "EF", "EC", "ETN", "ETM", "XPN", "XSN", "XSV", "XSA", "XR", "MAG", "MAJ", "IC", "SF", "SE", "SS", "SP", "SO", "SW", "SH", "SL", "SN", "NF", "NV", "NA"] #Sejong POS tagset

print(classification_report(test_tag_only,pt_tagger_tagged_tag_only,labels = xpos_list)) #calculate score






###################### Perceptron tagger: UPOS tagging ######################


## Machine-learning part
# average perceptron (by Matthew Honnibal) #https://explosion.ai/blog/part-of-speech-pos-tagger-in-python

class AveragedPerceptron_MorphemeStagUtag(object):

    def __init__(self):
        self.weights = {} #each feature gets its own weight vector; weights = dict-of-dicts
        self.classes = set()
        self._totals = defaultdict(int) #the accumulated values (for the averaging); keyed by feature/class tuples
        self._tstamps = defaultdict(int) #the last time the feature was changed (for the averaging); keyed by feature/class tuples; tstamps =  timestamps
        self.i = 0 #number of instances seen

    def predict(self, features):
        '''Dot-product the features and current weights and return the best label.'''
        scores = defaultdict(float) #
        for feat, value in features.items(): #iterate through features
            if feat not in self.weights or value == 0: #feature not found or 0 value > continue to next iteration
                continue
            weights = self.weights[feat] #feature in self.weights as weights
            for label, weight in weights.items(): #iterate through weights
                scores[label] += value * weight #calcualte score of each label

        return max(self.classes, key=lambda label: (scores[label], label)) #do secondary alphabetic sort for stability

    def update(self, truth, guess, features):
        '''Update the feature weights.'''
        def upd_feat(c, f, w, v): #c=class, f=feature, w=weight, v=vlaue
            param = (f, c) #feature-class pair
            self._totals[param] += (self.i - self._tstamps[param]) * w #calcualte weight for each pair
            self._tstamps[param] = self.i #???
            self.weights[f][c] = w + v #???

        self.i += 1 #increase self.i
        if truth == guess: #two weights equal > fine
            return None
        for f in features: #iterate through features
            weights = self.weights.setdefault(f, {}) #default weight
            upd_feat(truth, f, weights.get(truth, 0.0), 1.0) #update weight for feature: truth+1
            upd_feat(guess, f, weights.get(guess, 0.0), -1.0) #update weight for feature: guess-1

    def average_weights(self):
        '''Average weights from all iterations.'''
        for feat, weights in self.weights.items(): #iterate through weights.items
            new_feat_weights = {} #holder (dictionary) for average weight of feature
            for clas, weight in weights.items(): #iterate through weights.items
                param = (feat, clas) #feature-class pair
                total = self._totals[param] #total weight of each pair
                total += (self.i - self._tstamps[param]) * weight #calcualte weight for each pair
                averaged = round(total / self.i, 3) #calcualte average weight
                if averaged:
                    new_feat_weights[clas] = averaged #update weight
            self.weights[feat] = new_feat_weights #assign new weight to feature

    def save(self, path):
        '''Save the pickled model weights.'''
        with open(path, 'wb') as fout: #convention for saving pickle #wb = write byte
            return pickle.dump(dict(self.weights), fout)

    def load(self, path):
        '''Load the pickled model weights.'''
        self.weights = load(path) #convention for loading pickle


## Tagger part
# Greedy Averaged Perceptron tagger by Matthew Honnibal
# https://explosion.ai/blog/part-of-speech-pos-tagger-in-python

class PerceptronTagger_simple_MorphemeStagUtag():

    START = ['-START-', '-START2-']
    END = ['-END-', '-END2-']

    def __init__(self, load=True):
        '''
        :param load: Load the pickled model upon instantiation.
        '''
        self.model = AveragedPerceptron() #load AveragePerceptron class
        self.tagdict = {} #holder (dictionary)
        self.classes = set()
        if load:
            AP_MODEL_LOC = PICKLE #specify location of model
            self.load(AP_MODEL_LOC) #load model

    def tag(self, tokens):
        '''
        Tag tokenized sentences.
        :params tokens: list of word
        :type tokens: list(str)
        '''
        prev, prev2 = self.START #set prev & prev2
        post, post2 = self.END #set post & post2
        output = [] #holder for output

        context = self.START + [self.normalize(w) for w in tokens] + self.END #set context using envorinment of start & end and normalised weight for each word token
        for i, word in enumerate(tokens): #iterate through tokens #enumerate(): add index; here, index = i & item = word
            tag = self.tagdict.get(word) #tag as tag of word from tagdict
            if not tag: #not found in tagdict > find/predict the best tag based on probability
                features = self._get_features(i, word, context, prev, prev2, post, post2)
                tag = self.model.predict(features)
            output.append((word, tag)) #append predicted word-tag pair
            prev2 = prev #sequence +1
            prev = tag #sequence +1

        return output

    def train(self, sentences, save_loc=None, nr_iter=10):
        '''Train a model from sentences, and save it at ``save_loc``.
        ``nr_iter`` controls the number of Perceptron training iterations.
        :param sentences: A list or iterator of sentences, where each sentence
            is a list of (words, tags) tuples.
        :param save_loc: If not ``None``, saves a pickled model in this location.
        :param nr_iter: Number of training iterations.
        '''
        self._sentences = list()  #to be populated by self._make_tagdict
        self._make_tagdict(sentences) #create tag dictionary using sentences
        self.model.classes = self.classes
        for iter_ in range(nr_iter): #iterate through as many as number of iteration
            c = 0 #starting point for class
            n = 0 #starting point for number (of iteration)
            for sentence in self._sentences: #iterate through self._sentences
                words, tags = zip(*sentence) # *** THIS LINE SEEMS PROBLEMATIC... *** #
                prev, prev2 = self.START #set prev & prev2
                post, post2 = self.END
                context = self.START + [self.normalize(w) for w in words] + self.END #set context using envorinment of start & end and normalised weight for each word token
                for i, word in enumerate(words): #iterate through tokens #enumerate(): add index; here, index = i & item = word
                    guess = self.tagdict.get(word) #guess as tag of word from tagdict
                    if not guess: #not found in guess > find/predict the best tag based on probability > update info
                        feats = self._get_features(i, word, context, prev, prev2, post, post2)
                        guess = self.model.predict(feats)
                        self.model.update(tags[i], guess, feats)
                    prev2 = prev #sequence +1
                    prev = guess #sequence +1
                    c += guess == tags[i] #???
                    n += 1 #increase number of n(iteration)
            random.shuffle(self._sentences) #???
            logging.info("Iter {0}: {1}/{2}={3}".format(iter_, c, n, _pc(c, n))) #???

        self._sentences = None #delete training sentences (We don't need the training sentences anymore, and we don't want to waste space on them when we pickle the trained tagger)

        self.model.average_weights() #calculate average weight for feature in model

        if save_loc is not None: #parameter changed
            with open(save_loc, 'wb') as fout: #convention for saving pickle #wb = write byte
                pickle.dump((self.model.weights, self.tagdict, self.classes), fout, 2)
                #cf) changed protocol from -1 to 2 to make pickling Python 2 compatible

    def load(self, loc):
        '''
        :param loc: Load a pickled model at location.
        :type loc: str
        '''
        self.model.weights, self.tagdict, self.classes = pickle.load(open(loc,"rb"))
        self.model.classes = self.classes


    def normalize(self, word):
        '''
        Normalisation for pre-processing.
        - All words are lower cased
        - Groups of digits of length 4 are represented as !YEAR;
        - Other digits are represented as !DIGITS
        :rtype: str
        '''
        if '-' in word and word[0] != '-':
            return '!HYPHEN'
        elif word.isdigit() and len(word) == 4:
            return '!YEAR'
        elif word[0].isdigit():
            return '!DIGITS'
        else:
            return word.lower()


    def _get_features(self, i, word, context, prev, prev2, post, post2): #word: individual Stag_idx pair (split-able), context: the whole Stag_idx pair (not split-able), prev&post: Utag
        '''Map tokens into a feature representation, implemented as a
        {hashable: int} dict. If the features change, a new model must be
        trained.
        '''
        def add(name, *args): #function for adding feature-explanation-occurrence info
            features[' '.join((name,) + tuple(args))] += 1

        i += len(self.START)
        features = defaultdict(int)
        add('bias') #act like a prior

        # basic element in single word
        add('i morpheme combi', word.split("/")[0]) #morpheme_index in current combi
        add('i Stag combi', word.split("/")[1]) #Stag_index in current combi

        # context only
        add('i morpheme-Stag combi', context[i]) #current morpheme-Stag combi
        add('i-1 morpheme-Stag combi', context[i-1]) #previous morpheme-Stag combi
#        add('i-2 morpheme-Stag combi', context[i-2]) #morpheme-Stag combi before previous Stag combi
        add('i+1 morpheme-Stag combi', context[i+1]) #next morpheme-Stag combi
#        add('i+2 morpheme-Stag combi', context[i+2]) #morpheme-Stag combi next to next morpheme-Stag combi
##        add('i-2 morpheme-Stag combi + i-1 morpheme-Stag combi', context[i-2], context[i-1])
        add('i-1 morpheme-Stag combi + i morpheme-Stag combi', context[i-1], context[i])
        add('i morpheme-Stag combi + i+1 morpheme-Stag combi', context[i], context[i+1])
##        add('i+1 morpheme-Stag combi + i+2 morpheme-Stag combi', context[i+1], context[i+2])
#        add('i-2 morpheme-Stag combi + i-1 morpheme-Stag combi + i morpheme-Stag combi', context[i-2], context[i-1], context[i])
        add('i-1 morpheme-Stag combi + i morpheme-Stag combi + i+1 morpheme-Stag combi', context[i-1], context[i], context[i+1])
#        add('i morpheme-Stag combi + i+1 morpheme-Stag combi + i+2 morpheme-Stag combi', context[i], context[i+1], context[i+2])
##        add('i-2 morpheme-Stag combi + i-1 morpheme-Stag combi + i morpheme-Stag combi + i+1 morpheme-Stag combi', context[i-2], context[i-1], context[i], context[i+1])
##        add('i-1 morpheme-Stag combi + i morpheme-Stag combi + i+1 morpheme-Stag combi + i+2 morpheme-Stag combi', context[i-1], context[i], context[i+1], context[i+2])
        add('i-2 morpheme-Stag combi + i-1 morpheme-Stag combi + i morpheme-Stag combi + i+1 morpheme-Stag combi + i+2 morpheme-Stag combi', context[i-2], context[i-1], context[i], context[i+1], context[i+2])

        # Utag only
        add('i-1 Utag', prev) #Utag of previous Stag combi
#        add('i-2 Utag', prev2) #Utag of Stag combi before previous Stag combi
        add('i+1 Utag', post) #Utag of next Stag combi
#        add('i+2 Utag', post2) #Utag of Stag combi next to next Stag combi
        add('i-2 Utag + i-1 Utag', prev2, prev)  #Utag sequence of previous two words
        add('i+1 Utag + i+2 Utag', post, post2)  #Utag sequence of next two words
        add('i-1 Utag + i+1 Utag', prev, post)  #Utag sequence of next two words
##        add('i-2 Utag + i-1 Utag + i+1 Utag', prev2, prev, post)
##        add('i-1 Utag + i+1 Utag + i+2 Utag', prev, post, post2)
        add('i-2 Utag + i-1 Utag + i+1 Utag + i+2 Utag', prev2, prev, post, post2)

        # combination of context and Utag
        add('i-1 Utag + i morpheme-Stag combi', prev, context[i]) #bigram: Utag of previous Stag combi + current Stag combi
        add('i morpheme-Stag combi + i+1 Utag', context[i], post) #bigram: current morpheme-Stag combi + Utag of next Stag combi
        add('i-1 Utag + i morpheme-Stag combi + i+1 Utag', prev, context[i], post)
        add('i-1 morpheme-Stag combi + i-1 Utag + i morpheme-Stag combi', context[i-1], prev, context[i]) #trigram: previous morpheme-Stag combi + previous Utag + current morpheme-Stag combi
        add('i morpheme-Stag combi + i+1 morpheme-Stag combi + i+1 Utag', context[i], context[i+1], post) #trigram: current morpheme-Stag combi + next morpheme-Stag combi + next Utag
##        add('i-1 morpheme-Stag combi + i-1 Utag + i morpheme-Stag combi + i+1 Utag', context[i-1], prev, context[i], post) 
##        add('i-1 Utag + i morpheme-Stag combi + i+1 morpheme-Stag combi + i+1 Utag', prev, context[i], context[i+1], post)
        add('i-1 morpheme-Stag combi + i-1 Utag + i morpheme-Stag combi + i+1 morpheme-Stag combi + i+1 Utag', context[i-1], prev, context[i], context[i+1], post)
##        add('i-2 Utag + i-1 Utag + i morpheme-Stag combi', prev2, prev, context[i]) #trigram: Utag of previous two morpheme-Stag combi + current morpheme-Stag combi
##        add('i morpheme-Stag combi + i+1 Utag + i+2 Utag', context[i], post, post2) #trigram: current morpheme-Stag combi + Utag of following two Stag combi
#        add('i-2 Utag + i-1 Utag + i morpheme-Stag combi + i+1 Utag', prev2, prev, context[i], post)
#        add('i-1 Utag + i morpheme-Stag combi + i+1 Utag + i+2 Utag', prev, context[i], post, post2)
        add('i-2 Utag + i-1 Utag + i morpheme-Stag combi + i+1 Utag + i+2 Utag', prev2, prev, context[i], post, post2)

        # combination of morpheme combi and Utag
        add('i-1 Utag + i morpheme combi', prev, word.split("/")[0]) #bigram: Utag of previous Stag combi + 1st Stag_index in current combi
        add('i morpheme combi + i+1 Utag', word.split("/")[0], post) #bigram: 1st Stag_index in current combi + Utag of next Stag combi
#        add('i-2 Utag + i-1 Utag + i morpheme combi', prev2, prev, word.split("/")[0]) #trigram: Utag of previous two Stag combi + st Stag_index in current combi
#        add('i morpheme combi + i+1 Utag + i+2 Utag', word.split("/")[0], post, post2) #trigram: 1st Stag_index in current combi + Utag of following two Stag combi
        add('i-1 Utag + i morpheme combi + i+1 Utag', prev, word.split("/")[0], post)
##        add('i-2 Utag, i-1 Utag + i morpheme combi + i+1 Utag', prev2, prev, word.split("/")[0], post)
##        add('i-1 Utag + i morpheme combi + i+1 Utag + i+2 Utag', prev, word.split("/")[0], post, post2)
        add('i-2 Utag + i-1 Utag + i morpheme combi + i+1 Utag + i+2 Utag', prev2, prev, word.split("/")[0], post, post2)

        # combination of Stag combi and Utag
        add('i-1 Utag + i Stag combi', prev, word.split("/")[1]) #bigram: Utag of previous Stag combi + 1st Stag_index in current combi
        add('i Stag combi + i+1 Utag', word.split("/")[1], post) #bigram: 1st Stag_index in current combi + Utag of next Stag combi
#        add('i-2 Utag + i-1 Utag + i Stag combi', prev2, prev, word.split("/")[1]) #trigram: Utag of previous two Stag combi + st Stag_index in current combi
#        add('i Stag combi + i+1 Utag + i+2 Utag', word.split("/")[1], post, post2) #trigram: 1st Stag_index in current combi + Utag of following two Stag combi
        add('i-1 Utag + i Stag combi + i+1 Utag', prev, word.split("/")[1], post)
##        add('i-2 Utag, i-1 Utag + i Stag combi + i+1 Utag', prev2, prev, word.split("/")[1], post)
##        add('i-1 Utag + i Stag combi + i+1 Utag + i+2 Utag', prev, word.split("/")[1], post, post2)
        add('i-2 Utag + i-1 Utag + i Stag combi + i+1 Utag + i+2 Utag', prev2, prev, word.split("/")[1], post, post2)

        # combination of morpheme combi, Stag combi, and Utag
        add('i-1 morpheme-Stag + i-1 Utag + i morpheme combi', context[i-1], prev, word.split("/")[0]) #trigram: previous morpheme-Stag combi + previous Utag + 1st Stag_index in current combi
        add('i morpheme combi + i+1 morpheme-Stag + i+1 Utag', word.split("/")[0], context[i+1], post) #trigram: 1st Stag_index in current combi + next morpheme-Stag combi + next Utag
        add('i-1 morpheme-Stag + i-1 Utag + i Stag combi', context[i-1], prev, word.split("/")[1]) #trigram: previous morpheme-Stag combi + previous Utag + 1st Stag_index in current combi
        add('i Stag combi + i+1 morpheme-Stag + i+1 Utag', word.split("/")[1], context[i+1], post) #trigram: 1st Stag_index in current combi + next morpheme-Stag combi + next Utag
#        add('i-2 morpheme-Stag + i-2 Utag + i-1 morpheme-Stag + i-1 Utag + i morpheme combi', context[i-2], prev2, context[i-1], prev, word.split("/")[0])
#        add('i morpheme combi + i+1 morpheme-Stag + i+1 Utag + i+2 morpheme-Stag + i+2 Utag', word.split("/")[0], context[i+1], post, context[i+2], post2)
        add('i-1 morpheme-Stag + i-1 Utag + i morpheme combi + i+1 morpheme-Stag + i+1 Utag', context[i-1], prev, word.split("/")[0], context[i+1], post)
        add('i-2 morpheme-Stag + i-2 Utag + i-1 morpheme-Stag + i-1 Utag + i morpheme combi + i+1 morpheme-Stag + i+1 Utag + i+2 morpheme-Stag + i+2 Utag', context[i-2], prev2, context[i-1], prev, word.split("/")[0], context[i+1], post, context[i+2], post2)
#        add('i-2 morpheme-Stag + i-2 Utag + i-1 morpheme-Stag + i-1 Utag + i Stag combi', context[i-2], prev2, context[i-1], prev, word.split("/")[1])
#        add('i Stag combi + i+1 morpheme-Stag + i+1 Utag + i+2 morpheme-Stag + i+2 Utag', word.split("/")[1], context[i+1], post, context[i+2], post2)
        add('i-1 morpheme-Stag + i-1 Utag + i Stag combi + i+1 morpheme-Stag + i+1 Utag', context[i-1], prev, word.split("/")[1], context[i+1], post)
        add('i-2 morpheme-Stag + i-2 Utag + i-1 morpheme-Stag + i-1 Utag + i Stag combi + i+1 morpheme-Stag + i+1 Utag + i+2 morpheme-Stag + i+2 Utag', context[i-2], prev2, context[i-1], prev, word.split("/")[1], context[i+1], post, context[i+2], post2)


        return features


    def _make_tagdict(self, sentences):
        '''
        Make a tag dictionary for single-tag words.
        :param sentences: A list of list of (word, tag) tuples.
        '''
        counts = defaultdict(lambda: defaultdict(int)) #
        for sentence in sentences: #iterate through sentences
            self._sentences.append(sentence) #append sentence to self._sentences
            for word, tag in sentence: #iterate through sentence
                counts[word][tag] += 1 #increase number
                self.classes.add(tag) #add tag to classes
        freq_thresh = 20 #cut-off level for frequency
        ambiguity_thresh = 0.985 #cut-off level for determination of ambiguity
        for word, tag_freqs in counts.items(): #iterate through counts.items
            tag, mode = max(tag_freqs.items(), key=lambda item: item[1]) #???
            n = sum(tag_freqs.values()) #n as sum of freq value
            if n >= freq_thresh and (mode / n) >= ambiguity_thresh: #don't add rare words to the tag dictionary & only add unambiguous words
                self.tagdict[word] = tag

def _pc(n, d): #???
    return (n / d) * 100


if __name__ == '__main__':
    #_get_pretrain_model()
    pass


## train the tagger

with open("pretty_corpus_CHILDES_upos_tagging_20191009.pkl","rb") as infile: #convention for loading pickle #rb = read byte
    whole_data_upos = pickle.load(infile)

training = simple_train_test(whole_data_upos,.9)[0]
test = simple_train_test(whole_data_upos,.9)[1]
#dvlp_test = simple_train_test(whole_data,8.)[1]
#development = simple_train_test(dvlp_test,.5)[0]
#test = simple_train_test(dvlp_test,.5)[1]

#test_sents_dvlp = tag_strip_sents(development)
test_sents_test = tag_strip_sents(test)

print(training[-1]) #works fine #no overlap
print(len(training)) #works fine #62548
#print(dvlp_test[:10]) #works fine
#print(len(dvlp_test)) #works fine
#print(development[:5]) #works fine #no overlap
#print(len(development)) #works fine
print(test[-1]) #works fine #no overlap #[[(word, tag), ...], [(word, tag), ...]]
print(len(test)) #works fine #6950
#print(test_sents_dvlp[:5]) #works fine
#print(len(test_sents_dvlp)) #works fine
print(test_sents_test[-1]) #works fine; [[word, ...], [word, ...]]
print(len(test_sents_test)) #works fine #6950


## Accuracy, precision, recall, & F1 value for xpos tagging

# Define and train tagger

pt_tagger_upos = PerceptronTagger_simple_MorphemeStagUtag(load=False) #define tagger #load=False: do not lode the default model (i.e., pre-trained model) b/c we are going to train our model and use it
pt_tagger_upos.train(training,save_loc = "CHILDES_upos_perceptron_iter10_freq20_ambi985_20191009.pkl") #train and save model

#pt_tagger.load("training_feature_perceptron.pkl")


# Tag sentences (the tagger tags one sentence by default)

#pt_tagger_tagged_dvlp = []
#for x in test_sents_dvlp:
#    pt_tagger_tagged_dvlp.append(pt_tagger.tag(x))

pt_tagger_tagged_test = []
for x in test_sents_test:
    pt_tagger_tagged_test.append(pt_tagger_upos.tag(x))


# Check accuracy
    
#print(simple_accuracy_sent(development,pt_tagger_tagged_dvlp)) 
print(simple_accuracy_sent(test,pt_tagger_tagged_test)) #98.95%


# Precision, recall, f1-score

test_tag_only = tag_for_calculation(test) #de-tach tag from word-tag pair
print(test_tag_only[:10]) #works fine
print(len(test_tag_only)) #works fine

pt_tagger_tagged_tag_only = tag_for_calculation(pt_tagger_tagged_test) #de-tach tag from word-tag pair
print(pt_tagger_tagged_tag_only[:10]) #works fine
print(len(pt_tagger_tagged_tag_only)) #works fine

upos_list = ["NOUN", "PROPN", "PRON", "VERB", "ADJ", "ADV", "DET", "NUM", "CCONJ", "ADP", "PUNCT", "AUX", "INTJ", "PART", "SCONJ", "SYM", "X"] #UPOS tagset

print(classification_report(test_tag_only,pt_tagger_tagged_tag_only,labels = upos_list)) #calculate score





###################### Pattern-finding ######################
### four caveats
### 1. hard to extract patterns in complex clause
### 2. hard to consider ellipsis e.g., argument / case marker omission, rel
### 3. hard to detect (suffixal) passives / (morphological) causatives
### 4. tricky to distinguish canonical from scrambled (but possible by using numeric location as string)


## load pkg

from collections import Counter #creates frequency list efficiently


##### ***** issue: automatic file name not applied to output ***** #####
### https://realpython.com/python-f-strings/


## define function for outputting sents as txt file

def output_sent_txt(pattern_sent_list):

#    outf = open("CHILDES_freq/{}.txt".format(pattern_sent_list), "w")
    outf = open("CHILDES_freq/sent_result.txt", "w")

    for sent in pattern_sent_list:
        sent = sent + "\n"
        outf.write(sent)

    outf.flush() 
    outf.close()


## define function for verb freq list

def freq_list_verb(pattern_sent_list):

    verb_list = []
    
    for sent_for_freq in pattern_sent_list:
        eojeol_for_freq = sent_for_freq.split(" ")
        for eojeol_from_sentence in eojeol_for_freq:
            if "VV" in eojeol_from_sentence or "VX" in eojeol_from_sentence or "VERB" in eojeol_from_sentence:
                item_in_eojeol = eojeol_from_sentence.split("/")
                verb_from_item = item_in_eojeol[0]
                verb_list.append(verb_from_item)
    
    verb_freq = Counter(verb_list)
    verb_freq_sort = verb_freq.most_common()
    
    return(verb_freq_sort)


### define function for adj freq list (no need at this point)
#
#def freq_list_adj(pattern_sent_list):
#
#    adj_list = []
#    
#    for sent_for_freq in pattern_sent_list:
#        eojeol_for_freq = sent_for_freq.split(" ")
#        for eojeol_from_sentence in eojeol_for_freq:
#            if "VA" in eojeol_from_sentence or "ADJ" in eojeol_from_sentence:
#                item_in_eojeol = eojeol_from_sentence.split("/")
#                adj_from_item = item_in_eojeol[0]
#                adj_list.append(adj_from_item)
#    
#    adj_freq = Counter(adj_list)
#    adj_freq_sort = adj_freq.most_common()
#
#    return(adj_freq_sort)


## define function for outputting verb/adj freq list as txt file

def output_freq_list(freq_list):

#    outf = open("CHILDES_freq/%s.txt" % verb_freq_list, "w")
    outf = open("CHILDES_freq/freq_result.txt", "w")
    outf.write("item\tfrequency\n")
    
    for itemfreq_pair in freq_list:
        # commands reflecting the change of the data type: dictionary to tuple
        item = itemfreq_pair[0]
        freq = itemfreq_pair[1]
        # format for ouuput
        output = item + "\t" + str(freq) + "\n"
        outf.write(output)
    
    outf.flush() 
    outf.close()


## load data: conll format

text_conll_raw = open("CHILDES_final.txt").read().split("\n\n\n")
print(text_conll_raw[:3])
print(len(text_conll_raw))


## pre-processing (1): exclude single-morpheme utterance

text_conll = []

for sent_raw in text_conll_raw:
    if sent_raw == "\n":
        continue
    elif sent_raw == "":
        continue
    else:
        lines_raw = sent_raw.split("\n")
        line_fullsent = lines_raw[0]
        if len(line_fullsent) <= 15:
            continue
        else:
            text_conll.append(lines_raw)

print(text_conll[:3])
print(len(text_conll)) #69585


## pre-processing (2): merging info by sent

sents_for_pf_list = []

for sent in text_conll:
#    sent_for_pf_list = []
#    lines = sent.split("\n")
    eojeol_fullinfo_temp = []

    for line in sent:
        if line == "\n":
            continue
        elif line == "":
            continue
        elif "#" in line:
            continue
        else:
            items = line.split("\t")
            eojeol = items[1]
            eojeol_by_morph = items[2]
            upos = items[3]
            xpos = items[4]
#            dep = items[7]
#            eojeol_fullinfo = eojeol + "/" + eojeol_by_morph + "/" + xpos + "/" + dep + "/" + upos
            eojeol_fullinfo = eojeol + "/" + eojeol_by_morph + "/" + xpos + "/" + upos
            if "PUNCT" in eojeol_fullinfo:
                continue
            else:
                eojeol_fullinfo_temp.append(eojeol_fullinfo)
        
    sent_fullinfo = " ".join(eojeol_fullinfo_temp)
    sents_for_pf_list.append(sent_fullinfo)

print(sents_for_pf_list[-1]) #['기호는/기호+는/NNP+JX/NOUN 날마다/날+마다/NNG+JX/ADV 책을/책+을/NNG+JKO/NOUN 열심히/열심히/MAG/ADV 읽었어요/읽+었+어요/VV+EP+EF/VERB']
print(len(sents_for_pf_list))


sents_for_pf_list_sanity_chk = []

for sent in sents_for_pf_list:
    if sent == "":
        continue
    elif sent == " ":
        continue
    else:
        sents_for_pf_list_sanity_chk.append(sent)

print(sents_for_pf_list_sanity_chk[-1])
print(len(sents_for_pf_list_sanity_chk)) #69585





##### full active transitives ####

### canonical 
## morpheme: N-i/ka + N-(l)ul
## XPOS: NNG/NNP/NNB/NR/NP-JKS + NNG/NNP/NNB/NR/NP-JKO + VV
## UPOS: NOUN or PRON + NOUN or PRON + VERB

### scrambled
## morpheme: N-(l)ul + N-i/ka
## XPOS: NNG/NNP/NNB/NR/NP-JKO + NNG/NNP/NNB/NR/NP-JKS + VV
## UPOS: NOUN or PRON + NOUN or PRON + VERB


## extract sent by postposition: active transitives

# sent with JKS & JKO

sent_by_ppt_list = []

for sent in sents_for_pf_list_sanity_chk: #extract sent w/ two postpositions first
    if "VERB" in sent or "ADJ" in sent:
            if "JKS" in sent and "JKO"in sent:
                sent_by_ppt_list.append(sent)

#print(sent_by_ppt_list)
print(len(sent_by_ppt_list)) #2466


# sent by canonicity

canonical_active_transitive = []
scrambled_active_transitive = []

for sent in sent_by_ppt_list: #sort out cncl / scrm
    jks = sent.find("JKS")
    jko = sent.find("JKO")
#    print(jks, jko)
    if jks < jko:
        canonical_active_transitive.append(sent)
    else:
        scrambled_active_transitive.append(sent)

#print(canonical_active_transitive)
print(len(canonical_active_transitive)) #2002
#print(scrambled_active_transitive)
print(len(scrambled_active_transitive)) #464


## output as txt file

output_sent_txt(canonical_active_transitive)
output_sent_txt(scrambled_active_transitive)


## verb freq list

verb_list_cncl_act_tr = freq_list_verb(canonical_active_transitive)
output_freq_list(verb_list_cncl_act_tr)

verb_list_scrm_act_tr = freq_list_verb(scrambled_active_transitive)
output_freq_list(verb_list_scrm_act_tr)





##### full suffixal passives #####

### canonical
## morpheme: N-i/ka + N-eykey/hanthey + -i, -hi, -li, -ki in V
## XPOS: NNG/NNP/NNB/NR/NP-JKS + NNG/NNP/NNB/NR/NP-JKB + VV + XSV
## UPOS: NOUN or PRON + NOUN or PRON + VERB


### scrambled
## morpheme: N-eykey/hanthey + N-i/ka + -i, -hi, -li, -ki in V
## XPOS: NNG/NNP/NNB/NR/NP-JKB + NNG/NNP/NNB/NR/NP-JKS + VV + XSV
## UPOS: NOUN or PRON + NOUN or PRON + VERB


## extract sent by postposition & verbal morphology: suffixal passive

# sent with JKS & -eykey/hanthey as JKB

sent_by_ppt_list = []

for sent in sents_for_pf_list_sanity_chk: #extract sent w/ two postpositions first
    if "VERB" in sent or "ADJ" in sent:
        item_in_sent = sent.split(" ")
        for indiv_item in item_in_sent:
            if "XSV" in indiv_item:
                verb_morpheme = indiv_item.split("/")[1]
                if "이" in verb_morpheme or "히" in verb_morpheme or "리" in verb_morpheme or "기" in verb_morpheme:
                    sent_by_ppt_list.append(sent)

#print(sent_by_ppt_list)
print(len(sent_by_ppt_list)) #2480


# sent by canonicity

canonical_suffixal_passive = []
scrambled_suffixal_passive = []

for sent in sent_by_ppt_list: #sort out cncl / scrm
    jks = sent.find("JKS")
    jkb = sent.find("JKB")
#    print(jks, jko)
    if jks < jkb:
        canonical_suffixal_passive.append(sent)
    else:
        scrambled_suffixal_passive.append(sent)

#print(canonical_suffixal_passive)
print(len(canonical_suffixal_passive)) #663
#print(scrambled_suffixal_passive)
print(len(scrambled_suffixal_passive)) #1817


## output as txt file

output_sent_txt(canonical_suffixal_passive)
output_sent_txt(scrambled_suffixal_passive)


## verb freq list

verb_list_cncl_sfx_psv = freq_list_verb(canonical_suffixal_passive)
output_freq_list(verb_list_cncl_sfx_psv)

verb_list_scrm_sfx_psv = freq_list_verb(scrambled_suffixal_passive)
output_freq_list(verb_list_scrm_sfx_psv)





##### NOM-only patterns #####
### canonical active transitives, NOM only
## morpheme: N-i/ka + N
## XPOS: NNG/NNP/NNB/NR/NP-JKS + NNG/NNP/NNB/NR/NP + VV
## UPOS: NOUN or PRON + NOUN or PRON + VERB

### scrambled active transitives, NOM only
## morpheme: N + N-i/ka
## XPOS: NNG/NNP/NNB/NR/NP + NNG/NNP/NNB/NR/NP-JKS + VV
## UPOS: NOUN or PRON + NOUN or PRON + VERB

### one-argument actives, NOM only ##
## morpheme: N-i/ka
## XPOS: NNG/NNP/NNB/NR/NP-JKS + VV
## UPOS: NOUN or PRON + VERB

### truncated suffixal passives: NOM only ##
## morpheme: N-i/ka + -i, -hi, -li, -ki in V
## XPOS: NNG/NNP/NNB/NR/NP-JKS + VV + XSV
## UPOS: NOUN or PRON + VERB



## extract sent by postposition

# sent with only JKS

sent_by_ppt_list = []
#postposition_list_without_jks = ["JKG", "JKO", "JKB", "JKV", "JKQ", "JC", "JX"]

for sent in sents_for_pf_list_sanity_chk: #sents only JKS
    if "VERB" in sent or "ADJ" in sent:
        if "JKS" in sent:
            if "JKO" not in sent and "JKB" not in sent and "JX" not in sent:
                sent_by_ppt_list.append(sent)

#print(sent_by_ppt_list)
print(len(sent_by_ppt_list)) #10224


# sent by arg#

one_arg_nom = []
two_arg_nom = []

for sent in sent_by_ppt_list: #count # of NOUN > sort out 1arg & 2arg
    count_noun = sent.count("NOUN")
    count_pron = sent.count("PRON")
    if count_noun == 1:
        one_arg_nom.append(sent)
    elif count_pron == 1:
        one_arg_nom.append(sent)
    else:
        two_arg_nom.append(sent)

#print(one_arg_nom)
print(len(one_arg_nom)) #5900
#print(two_arg_nom)
print(len(two_arg_nom)) #4324


# 1-arg sent by voice

one_arg_nom_psv = []

for sent in one_arg_nom:
    item_in_sent = sent.split(" ")
    for indiv_item in item_in_sent:
        if "XSV" in indiv_item:
            verb_morpheme = indiv_item.split("/")[1]
            if "이" in verb_morpheme or "히" in verb_morpheme or "리" in verb_morpheme or "기" in verb_morpheme:
                one_arg_nom_psv.append(sent)

#print(one_arg_nom_psv)
print(len(one_arg_nom_psv)) #225

one_arg_nom_act = [x for x in one_arg_nom if x not in one_arg_nom_psv]
#print(one_arg_nom_act)
print(len(one_arg_nom_act)) #5678


# 2-arg sent by voice

two_arg_nom_psv = []

for sent in two_arg_nom:
    item_in_sent = sent.split(" ")
    for indiv_item in item_in_sent:
        if "XSV" in indiv_item:
            verb_morpheme = indiv_item.split("/")[1]
            if "이" in verb_morpheme or "히" in verb_morpheme or "리" in verb_morpheme or "기" in verb_morpheme:
                two_arg_nom_psv.append(sent)

#print(two_arg_nom_psv)
print(len(two_arg_nom_psv)) #190

two_arg_nom_act = [x for x in two_arg_nom if x not in two_arg_nom_psv]
#print(two_arg_nom_act)
print(len(two_arg_nom_act)) #4140


# 2-arg active transitive by canonicity

canonical_active_transitive_nom = []

for sent in two_arg_nom_act: #sort out cncl / scrm #intuition: jks prior to any NOUN in cncl, & NOUN prior to jks in scrm
    jks = sent.find("JKS")
    noun_without_ppt = sent.find("NOUN")
    pron_without_ppt = sent.find("PRON")
#    print(jks, noun_without_ppt)
    if jks < noun_without_ppt:
        canonical_active_transitive_nom.append(sent)
    elif jks < pron_without_ppt:
        canonical_active_transitive_nom.append(sent)

#print(canonical_active_transitive_nom)
print(len(canonical_active_transitive_nom)) #1655

scrambled_active_transitive_nom = [x for x in two_arg_nom_act if x not in canonical_active_transitive_nom]
#print(scrambled_active_transitive_nom)
print(len(scrambled_active_transitive_nom)) #2485


# 2-arg suffixal passive by canonicity

canonical_suffixal_passive_nom = []

for sent in two_arg_nom_psv: #sort out cncl / scrm #intuition: jks prior to any NOUN in cncl, & NOUN prior to jks in scrm
    jks = sent.find("JKS")
    noun_without_ppt = sent.find("NOUN")
    pron_without_ppt = sent.find("PRON")
#    print(jks, noun_without_ppt)
    if jks < noun_without_ppt:
        canonical_suffixal_passive_nom.append(sent)
    if jks < pron_without_ppt:
        canonical_suffixal_passive_nom.append(sent)

#print(canonical_suffixal_passive_nom)
print(len(canonical_suffixal_passive_nom)) #68

scrambled_suffixal_passive_nom = [x for x in two_arg_nom_psv if x not in canonical_suffixal_passive_nom]
#print(scrambled_suffixal_passive_nom)
print(len(scrambled_suffixal_passive_nom)) #122


## output as txt file

output_sent_txt(one_arg_nom_act)
output_sent_txt(one_arg_nom_psv)
output_sent_txt(canonical_active_transitive_nom)
output_sent_txt(scrambled_active_transitive_nom)
output_sent_txt(canonical_suffixal_passive_nom)
output_sent_txt(scrambled_suffixal_passive_nom)


## verb freq list

verb_list_one_arg_nom_act = freq_list_verb(one_arg_nom_act)
output_freq_list(verb_list_one_arg_nom_act)

verb_list_one_arg_nom_psv = freq_list_verb(one_arg_nom_psv)
output_freq_list(verb_list_one_arg_nom_psv)

verb_list_cncl_act_tr_nom = freq_list_verb(canonical_active_transitive_nom)
output_freq_list(verb_list_cncl_act_tr_nom)

verb_list_scrm_act_tr_nom = freq_list_verb(scrambled_active_transitive_nom)
output_freq_list(verb_list_scrm_act_tr_nom)

verb_list_cncl_sfx_psv_nom = freq_list_verb(canonical_suffixal_passive_nom)
output_freq_list(verb_list_cncl_sfx_psv_nom)

verb_list_scrm_sfx_psv_nom = freq_list_verb(scrambled_suffixal_passive_nom)
output_freq_list(verb_list_scrm_sfx_psv_nom)





##### ACC-only patterns #####
### canonical active transitives, ACC only
## morpheme: N + N-(l)ul
## XPOS: NNG/NNP/NNB/NR/NP + NNG/NNP/NNB/NR/NP-JKO + VV
## UPOS: NOUN or PRON + NOUN or PRON + VERB

### scrmabled active transitives, ACC only
## morpheme: N-(l)ul + N
## XPOS: NNG/NNP/NNB/NR/NP-JKO + NNG/NNP/NNB/NR/NP + VV
## UPOS: NOUN or PRON + NOUN or PRON + VERB

### one-argument actives, ACC only ##
## morpheme: N-(l)ul
## XPOS: NNG/NNP/NNB/NR/NP-JKO + VV
## UPOS: NOUN or PRON + VERB



## extract sent by postposition

# sent with only JKO

sent_by_ppt_list = []

for sent in sents_for_pf_list_sanity_chk: #sents only JKO
    if "VERB" in sent:
        if "JKO" in sent:
            if  "JKS" not in sent and "JKB" not in sent and "JX" not in sent:
                sent_by_ppt_list.append(sent)

#print(sent_by_ppt_list)
print(len(sent_by_ppt_list)) #3658


# sent by arg#

one_arg_acc = []
two_arg_acc = []

for sent in sent_by_ppt_list: #count # of NOUN > sort out 1arg & 2arg
    count_noun = sent.count("NOUN")
    count_pron = sent.count("PRON")
    if count_noun == 1:
        one_arg_acc.append(sent)
    elif count_pron == 1:
        one_arg_acc.append(sent)
    else:
        two_arg_acc.append(sent)

#print(one_arg_acc)
print(len(one_arg_acc)) #2139
#print(two_arg_acc)
print(len(two_arg_acc)) #1519


# 2-arg sent by canonicity #caveat: NOUN but not subject possible in cncl e.g., '무릎/무릎/NNG/flat/NOUN 선을/선+을/NNG+JKO/flat/NOUN 살짝/살짝/MAG/flat/ADV 가리는/가리+는/VV+ETM/flat/VERB'

scrambled_active_transitive_acc = []

for sent in two_arg_acc: #sort out cncl / scrm #intuition: jko prior to any NOUN in scrm, & NOUN prior to jko in cncl
    jko = sent.find("JKO")
    noun_without_ppt = sent.find("NOUN")
    pron_without_ppt = sent.find("PRON")
#    print(jko, noun_without_ppt)
    if jko < noun_without_ppt:
        scrambled_active_transitive_acc.append(sent)
    elif jko < pron_without_ppt:
        scrambled_active_transitive_acc.append(sent)

#print(scrambled_active_transitive_acc)
print(len(scrambled_active_transitive_acc)) #455

canonical_active_transitive_acc = [x for x in two_arg_acc if x not in scrambled_active_transitive_acc]
#print(canonical_active_transitive_acc)
print(len(canonical_active_transitive_acc)) #1064


## output as txt file

output_sent_txt(one_arg_acc)
output_sent_txt(canonical_active_transitive_acc)
output_sent_txt(scrambled_active_transitive_acc)


## verb freq list

verb_list_act_acc = freq_list_verb(one_arg_acc)
output_freq_list(verb_list_act_acc)

verb_list_cncl_act_tr_acc = freq_list_verb(canonical_active_transitive_acc)
output_freq_list(verb_list_cncl_act_tr_acc)

verb_list_scrm_act_tr_acc = freq_list_verb(scrambled_active_transitive_acc)
output_freq_list(verb_list_scrm_act_tr_acc)





##### DAT-only patterns #####
### canonical suffixal passives, DAT only
## morpheme: N + N-eykey/hanthey/kkey + -i, -hi, -li, -ki in V
## XPOS: NNG/NNP/NNB/NR/NP + NNG/NNP/NNB/NR/NP-JKB + VV + XSV
## UPOS: NOUN or PRON + NOUN or PRON + VERB

### scrmabled suffixal passives, DAT only
## morpheme: N-eykey/hanthey/kkey + N + -i, -hi, -li, -ki in V + XSV
## XPOS: NNG/NNP/NNB/NR/NP-JKB + NNG/NNP/NNB/NR/NP + VV
## UPOS: NOUN or PRON + NOUN or PRON + VERB

### one-argument suffixal passives, DAT only ##
## morpheme: N-eykey/hanthey/kkey + -i, -hi, -li, -ki in V
## XPOS: NNG/NNP/NNB/NR/NP-JKB + VV + XSV
## UPOS: NOUN or PRON + VERB



## extract sent by postposition

# sent with only JKB

sent_by_ppt_list = []

for sent in sents_for_pf_list_sanity_chk: #sents only 에게/한테
    if "에게" in sent or "한테" in sent or "께"in sent:
        if "JKG" not in sent and "JKS" not in sent and "JKO" not in sent and "JX" not in sent:
            sent_by_ppt_list.append(sent)
        else:
            continue
    else:
        continue

#print(sent_by_ppt_list)
print(len(sent_by_ppt_list)) #418


# sent by arg#

one_arg_dat = []
two_arg_dat = []

for sent in sent_by_ppt_list: #count # of NOUN > sort out 1arg & 2arg
    count_noun = sent.count("NOUN")
    count_pron = sent.count("PRON")
    if count_noun == 1:
        one_arg_dat.append(sent)
    elif count_pron == 1:
        one_arg_dat.append(sent)
    else:
        two_arg_dat.append(sent)

#print(one_arg_dat)
print(len(one_arg_dat)) #147
#print(two_arg_dat)
print(len(two_arg_dat)) #271


# 1-arg sent by voice

one_arg_dat_act = []
one_arg_dat_psv = []

for sent in one_arg_dat:
    item_in_sent = sent.split(" ")
    for indiv_item in item_in_sent:
        if "VERB" in indiv_item:
            verb_morpheme = indiv_item.split("/")[1]
            if "XSV" in verb_morpheme:
                if "이" in verb_morpheme or "히" in verb_morpheme or "리" in verb_morpheme or "기" in verb_morpheme:
                    one_arg_dat_psv.append(sent)
                else:
                    one_arg_dat_act.append(sent)
            else:
                one_arg_dat_act.append(sent)
        else:
            continue

#print(one_arg_dat_act)
print(len(one_arg_dat_act)) #219
#print(one_arg_dat_psv)
print(len(one_arg_dat_psv)) #0


# 2-arg sent by voice

two_arg_dat_act = []
two_arg_dat_psv = []

for sent in two_arg_dat:
    item_in_sent = sent.split(" ")
    for indiv_item in item_in_sent:
        if "VERB" in indiv_item:
            verb_morpheme = indiv_item.split("/")[1]
            if "XSV" in verb_morpheme:
                if "이" in verb_morpheme or "히" in verb_morpheme or "리" in verb_morpheme or "기" in verb_morpheme:
                    two_arg_dat_psv.append(sent)
                else:
                    two_arg_dat_act.append(sent)
            else:
                two_arg_dat_act.append(sent)
        else:
            continue

#print(two_arg_dat_act)
print(len(two_arg_dat_act)) #390
#print(two_arg_dat_psv)
print(len(two_arg_dat_psv)) #0


# 2-arg active dative by canonicity

canonical_active_dat = []
scrambled_active_dat = []

for sent in two_arg_dat_act: #sort out cncl / scrm #intuition: jkb prior to any NOUN in scrm, & NOUN prior to jkb in cncl
    jkb = sent.find("JKB")
    noun_without_ppt = sent.find("NOUN")
    pron_without_ppt = sent.find("PRON")
#    print(jkb, noun_without_ppt)
    if jkb > noun_without_ppt:
        canonical_active_dat.append(sent)
    elif jkb > pron_without_ppt:
        canonical_active_dat.append(sent)
    else:
        scrambled_active_dat.append(sent)

#print(canonical_active_dat)
print(len(canonical_active_dat)) #338
#print(scrambled_active_dat)
print(len(scrambled_active_dat)) #52


# 2-arg suffixal passive by canonicity

canonical_suffixal_passive_dat = []
scrambled_suffixal_passive_dat = []

for sent in two_arg_dat_psv: #sort out cncl / scrm #intuition: jkb prior to any NOUN in scrm, & NOUN prior to jkb in cncl
    jkb = sent.find("JKB")
    noun_without_ppt = sent.find("NOUN")
    pron_without_ppt = sent.find("PRON")
#    print(jkb, noun_without_ppt)
    if jkb > noun_without_ppt:
        canonical_suffixal_passive_dat.append(sent)
    elif jkb > pron_without_ppt:
        canonical_suffixal_passive_dat.append(sent)
    else:
        scrambled_suffixal_passive_dat.append(sent)

#print(canonical_suffixal_passive_dat)
print(len(canonical_suffixal_passive_dat)) #0
#print(scrambled_suffixal_passive_dat)
print(len(scrambled_suffixal_passive_dat)) #0


## output as txt file

output_sent_txt(one_arg_dat_act)
output_sent_txt(one_arg_dat_psv)
output_sent_txt(canonical_active_dat)
output_sent_txt(scrambled_active_dat)
output_sent_txt(canonical_suffixal_passive_dat)
output_sent_txt(scrambled_suffixal_passive_dat)


## verb freq list

verb_list_one_arg_dat_act = freq_list_verb(one_arg_dat_act)
output_freq_list(verb_list_one_arg_dat_act)

verb_list_one_arg_dat_psv = freq_list_verb(one_arg_dat_psv)
output_freq_list(verb_list_one_arg_dat_psv)

verb_list_cncl_act_dat = freq_list_verb(canonical_active_dat)
output_freq_list(verb_list_cncl_act_dat)

verb_list_scrm_act_dat = freq_list_verb(scrambled_active_dat)
output_freq_list(verb_list_scrm_act_dat)

verb_list_cncl_sfx_psv_dat = freq_list_verb(canonical_suffixal_passive_dat)
output_freq_list(verb_list_cncl_sfx_psv_dat)

verb_list_scrm_sfx_psv_dat = freq_list_verb(scrambled_suffixal_passive_dat)
output_freq_list(verb_list_scrm_sfx_psv_dat)





##### patterns without involving case marking #####
### active transitives
## morpheme: N + N
## XPOS: NNG/NNP/NNB/NR/NP + NNG/NNP/NNB/NR/NP + VV
## UPOS: NOUN or PRON + NOUN or PRON + VERB

### suffixal passives
## morpheme: N + N + -i, -hi, -li, -ki in V
## XPOS: NNG/NNP/NNB/NR/NP + NNG/NNP/NNB/NR/NP + VV + XSV
## UPOS: NOUN or PRON + NOUN or PRON + VERB

### truncated actives
## morpheme: N
## XPOS: NNG/NNP/NNB/NR/NP + VV
## UPOS: NOUN or PRON + VERB

## truncated suffixal passives
## morpheme: N + -i, -hi, -li, -ki in V
## XPOS: NNG/NNP/NNB/NR/NP + VV + XSV
## UPOS: NOUN or PRON + VERB



## extract sent by postposition

# sent without case markers

sent_by_ppt_list = []
#postposition_list_without_jks = ["JKG", "JKO", "JKB", "JKV", "JKQ", "JC", "JX"]

for sent in sents_for_pf_list_sanity_chk: #sents without case markers
    if "NOUN" in sent or "PRON" in sent:
        if "JKS" not in sent and "JKG" not in sent and "JKO" not in sent and "JKB" not in sent and "JX" not in sent:
            sent_by_ppt_list.append(sent)
        else:
            continue
    else:
        continue

#print(sent_by_ppt_list)
print(len(sent_by_ppt_list)) #19418


# sent by arg#

one_arg = []
two_arg = []

for sent in sent_by_ppt_list: #count # of NOUN > sort out 1arg & 2arg
    count_noun = sent.count("NOUN")
    count_pron = sent.count("PRON")
    if count_noun == 1:
        one_arg.append(sent)
    elif count_pron == 1:
        one_arg.append(sent)
    else:
        two_arg.append(sent)

#print(one_arg)
print(len(one_arg)) #12803
#print(two_arg)
print(len(two_arg)) #6615


# 1-arg sent by voice

one_arg_act = []
one_arg_psv = []

for sent in one_arg:
    item_in_sent = sent.split(" ")
    for indiv_item in item_in_sent:
        if "VERB" in indiv_item:
            verb_morpheme = indiv_item.split("/")[1]
            if "XSV" in verb_morpheme:
                if "이" in verb_morpheme or "히" in verb_morpheme or "리" in verb_morpheme or "기" in verb_morpheme:
                    one_arg_psv.append(sent)
                else:
                    one_arg_act.append(sent)
            else:
                one_arg_act.append(sent)
        else:
            continue

#print(one_arg_act)
print(len(one_arg_act)) #15820
#print(one_arg_psv)
print(len(one_arg_psv)) #0


# 2-arg sent by voice

two_arg_act = []
two_arg_sfx_psv = []

for sent in two_arg:
    item_in_sent = sent.split(" ")
    for indiv_item in item_in_sent:
        if "VERB" in indiv_item:
            verb_morpheme = indiv_item.split("/")[1]
            if "XSV" in verb_morpheme:
                if "이" in verb_morpheme or "히" in verb_morpheme or "리" in verb_morpheme or "기" in verb_morpheme:
                    two_arg_sfx_psv.append(sent)
                else:
                    two_arg_act.append(sent)
            else:
                two_arg_act.append(sent)
        else:
            continue

#print(two_arg_act)
print(len(two_arg_act)) #7151
#print(two_arg_sfx_psv)
print(len(two_arg_sfx_psv)) #0


## output as txt file

output_sent_txt(one_arg_act)
output_sent_txt(one_arg_psv)
output_sent_txt(two_arg_act)
output_sent_txt(two_arg_sfx_psv)


## verb freq list

verb_list_one_arg_act = freq_list_verb(one_arg_dat_act)
output_freq_list(verb_list_one_arg_act)

verb_list_one_arg_psv = freq_list_verb(one_arg_psv)
output_freq_list(verb_list_one_arg_psv)

verb_list_two_arg_act = freq_list_verb(two_arg_act)
output_freq_list(verb_list_two_arg_act)

verb_list_two_arg_sfx_psv = freq_list_verb(two_arg_sfx_psv)
output_freq_list(verb_list_two_arg_sfx_psv)





##### lexical & periphrastic passives #####

### lexical passives

## extract sent by verb lemma

# sent with only passive verbs

sent_by_ppt_list = []

for sent in sents_for_pf_list_sanity_chk: #sents only passive verbs
    items = sent.split()
    for item in items:
        if "VERB" in item:
            verb_morpheme = item.split("/")[1]
            if "맞" in verb_morpheme or "받" in verb_morpheme or "빼앗기" in verb_morpheme or "당하" in verb_morpheme or "되" in verb_morpheme:
                sent_by_ppt_list.append(sent)
            else:
                continue
        else:
            continue
    else:
        continue

#print(sent_by_ppt_list)
print(len(sent_by_ppt_list)) #3197


# sent by arg#

one_arg_lxp = []
two_arg_lxp = []

for sent in sent_by_ppt_list: #count # of NOUN > sort out 1arg & 2arg
    count_noun = sent.count("NOUN")
    count_pron = sent.count("PRON")
    if count_noun == 1:
        one_arg_lxp.append(sent)
    elif count_pron == 1:
        one_arg_lxp.append(sent)
    else:
        two_arg_lxp.append(sent)

#print(one_arg_lxp)
print(len(one_arg_lxp)) #1354
#print(two_arg_lxp)
print(len(two_arg_lxp)) #1843


# 2-arg sent by canonicity 

canonical_lxp = []
scrambled_lxp = []

for sent in two_arg_lxp: #sort out cncl / scrm
    jks = sent.find("JKS")
    noun_without_ppt = sent.find("NOUN")
    pron_without_ppt = sent.find("PRON")
#    print(jks, noun_without_ppt)
    if jks < noun_without_ppt:
        canonical_lxp.append(sent)
    elif jks < pron_without_ppt:
        canonical_lxp.append(sent)
    else:
        scrambled_lxp.append(sent)

#print(canonical_lxp)
print(len(canonical_lxp)) #777
#print(scrambled_lxp)
print(len(scrambled_lxp)) #1066


## output as txt file

output_sent_txt(one_arg_lxp)
output_sent_txt(canonical_lxp)
output_sent_txt(scrambled_lxp)


## verb freq list

verb_list_one_arg_lxp = freq_list_verb(one_arg_lxp)
output_freq_list(verb_list_one_arg_lxp)

verb_list_canonical_lxp_list = freq_list_verb(canonical_lxp)
output_freq_list(verb_list_canonical_lxp_list)

verb_list_scrambled_lxp_list = freq_list_verb(scrambled_lxp)
output_freq_list(verb_list_scrambled_lxp_list)



### periphrastic passives

## extract sent by verb lemma

# sent with only passive verbs

sent_by_ppt_list = []

for sent in sents_for_pf_list_sanity_chk: #sents only passive verbs
    items = sent.split()
    for item in items:
        if "VERB" in item:
            if "어지" in item or "아지" in item:
                sent_by_ppt_list.append(sent)
            else:
                continue
        else:
            continue
    else:
        continue

#print(sent_by_ppt_list)
print(len(sent_by_ppt_list)) #857


# sent by arg#

one_arg_peri = []
two_arg_peri = []

for sent in sent_by_ppt_list: #count # of NOUN > sort out 1arg & 2arg
    count_noun = sent.count("NOUN")
    count_pron = sent.count("PRON")
    if count_noun == 1:
        one_arg_peri.append(sent)
    elif count_pron == 1:
        one_arg_peri.append(sent)
    else:
        two_arg_peri.append(sent)

#print(one_arg_peri)
print(len(one_arg_peri)) #372
#print(two_arg_peri)
print(len(two_arg_peri)) #485


# 2-arg sent by canonicity

canonical_peri = []
scrambled_peri = []

for sent in two_arg_peri: #sort out cncl / scrm
    jks = sent.find("JKS")
    noun_without_ppt = sent.find("NOUN")
    pron_without_ppt = sent.find("PRON")
#    print(jks, noun_without_ppt)
    if jks < noun_without_ppt:
        canonical_peri.append(sent)
    elif jks < pron_without_ppt:
        canonical_peri.append(sent)
    else:
        scrambled_peri.append(sent)

#print(canonical_peri)
print(len(canonical_peri)) #162
#print(scrambled_peri)
print(len(scrambled_peri)) #323


## output as txt file

output_sent_txt(one_arg_peri)
output_sent_txt(canonical_peri)
output_sent_txt(scrambled_peri)


## verb freq list

verb_list_one_arg_peri = freq_list_verb(one_arg_peri)
output_freq_list(verb_list_one_arg_peri)

verb_list_canonical_peri_list = freq_list_verb(canonical_peri)
output_freq_list(verb_list_canonical_peri_list)

verb_list_scrambled_peri_list = freq_list_verb(scrambled_peri)
output_freq_list(verb_list_scrambled_peri_list)





##### ADJ as predicate #####

### with NOM

## extract sent by postposition

# sent with only JKS

sent_by_ppt_list = []
#postposition_list_without_jks = ["JKG", "JKO", "JKB", "JKV", "JKQ", "JC", "JX"]

for sent in sents_for_pf_list_sanity_chk: #sents only JKS
    if "JKS" in sent:
        if "JKG" not in sent and "JKO" not in sent and "JKB" not in sent and "JX" not in sent:
            sent_by_ppt_list.append(sent)
        else:
            continue
    else:
        continue

#print(sent_by_ppt_list[-3:])
print(len(sent_by_ppt_list)) #10580


# sent by adj

nom_adj_combi = []

for sent in sent_by_ppt_list:
    item_in_sent = sent.split(" ")
    for indiv_item in item_in_sent:
        if "ADJ" in indiv_item: #VA not included b/c VA+XSV -> VERB
            nom_adj_combi.append(sent)
        else:
            continue

#print(nom_adj_combi[-3:])
print(len(nom_adj_combi)) #2797


# sanity chk

nom_adj = []

for sent in nom_adj_combi: 
    jks = sent.find("JKS")
    adj = sent.find("ADJ")
#    print(jks, adj)
    if jks > adj:
        continue
    else:
        nom_adj.append(sent)

#print(nom_adj[-3:])
print(len(nom_adj)) #1662


## output as txt file

output_sent_txt(nom_adj)



### without NOM

## extract sent by postposition

# sent without case markers

sent_by_ppt_list = []
#postposition_list_without_jks = ["JKG", "JKO", "JKB", "JKV", "JKQ", "JC", "JX"]

for sent in sents_for_pf_list_sanity_chk: #sents without case markers
    if "NOUN" in sent or "PRON" in sent:
        if "JKS" not in sent and "JKG" not in sent and "JKO" not in sent and "JKB" not in sent and "JX" not in sent:
            sent_by_ppt_list.append(sent)
        else:
            continue
    else:
        continue

#print(sent_by_ppt_list)
print(len(sent_by_ppt_list)) #19418


# sent by adj

noun_adj_combi = []

for sent in sent_by_ppt_list:
    item_in_sent = sent.split(" ")
    for indiv_item in item_in_sent:
        if "ADJ" in indiv_item:
            noun_adj_combi.append(sent)
        else:
            continue

#print(noun_adj_combi[-3:])
print(len(noun_adj_combi)) #3655


# sanity chk

noun_adj = []

for sent in noun_adj_combi: 
    noun = sent.find("NOUN")
    pron = sent.find("PRON")
    adj = sent.find("ADJ")
#    print(noun, adj)
    if noun > adj:
        continue
    elif pron > adj:
        continue
    else:
        noun_adj.append(sent)

#print(noun_adj[-3:])
print(len(noun_adj)) #2060


## output as txt file

output_sent_txt(noun_adj)





##### freq re-calcualtion after manual inspection #####


#text_corr = open("CHILDES_freq/corrected_sent/sent_cncl_act_tr_acc_corr_20190308.txt").read().split("\n")
#print(text_corr[:3])
#print(len(text_corr))
#
#
#verb_list = []
#
#for sent_for_freq in further_sanity:
#    eojeol_for_freq = sent_for_freq.split(" ")
#    for eojeol_from_sentence in eojeol_for_freq:
#        if "VV" in eojeol_from_sentence or "VX" in eojeol_from_sentence or "VERB" in eojeol_from_sentence:
#            item_in_eojeol = eojeol_from_sentence.split("/")
#            verb_from_item = item_in_eojeol[0]
#            verb_list.append(verb_from_item)
#        else:
#            continue
#
#verb_freq = Counter(verb_list)
#verb_freq_sort = verb_freq.most_common()
##print(verb_freq_sort)
#output_sent_verblist(verb_freq_sort)


#text_more = open("CHILDES_freq/sent_one_arg_acc.txt").read().split("\n")
#print(len(text_more))
#
#further_sanity = []
#
#for x in text_more:
#    y = x.split(" ")
#    for y_y in y:
#        if "NOUN" in y_y:
##            z = y_y.split("/")[1]
#            if "+이" in y_y or "+가" in y_y:
#                continue
#            else:
#                further_sanity.append(x)
#
#print(len(further_sanity))



#outf = open("CHILDES_freq/sent_scrm_act_tr_nom_without_existential_corr_20190308.txt", "w")
#
#for sent in one_arg_peri:
#    sent = sent + "\n"
#    outf.write(sent)
#
#outf.flush() 
#outf.close()

