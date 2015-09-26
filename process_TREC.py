import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
import imp
from preindex import node

def build_data_cv(file, split_dict, label_dict, clean_string=False):
    """
    Loads data and split data
    """
    revs = []
    f = open(file)
    vocab = defaultdict(float)
    
    for index, line in enumerate(f.readlines()):       
        rev = []
        rev.append(line.strip())
        if clean_string:
            orig_rev = clean_str(" ".join(rev))
        else:
            orig_rev = " ".join(rev)
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum  = {"y":label_dict[index], 
                    "text": orig_rev,                             
                    "num_words": len(orig_rev.split()),
                    "split": split_dict[index]}
        revs.append(datum)

    return revs, vocab
    
def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k))            
    W[0] = np.zeros(k)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs


def add_tree2vocab (sent, vocab):
    
    for j, each_word in enumerate(sent[:-1]):
        for l, each_field in enumerate(each_word):
            if each_field in vocab:
                continue
            elif each_field == 0:
                continue
            elif each_field == "ROOT":
                continue
            else:
                vocab[each_field] += 1


def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
        
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

def set_conv_sent(tree,labels_dict,max_len):
    conv_length = 5
    
    sent_num = len(tree)
    ##65-4 the most beginning 4 will be append to the front at last
    #sent_tensor = np.array.zeros((1,61,5))
    #sent_counter = 0
    doc_list =[]
    for ind,sents in enumerate(tree):
        sent_list = []
        for key in sents:
            if key == 0:
                continue
            currnet_node = sents[key]
            word_list = []
            for i in range(conv_length):
                if currnet_node.word != "ROOT":
                    word_list.append(currnet_node.word)
                else:
                    word_list.append(currnet_node.word)
                
                if currnet_node.word != "ROOT":
                    currnet_node = sents[currnet_node.parentindex]
            sent_list.append(word_list)
            
        header = []
        dummy = ["ROOT"]*conv_length
        for i in range(conv_length-1):
            header.append(dummy[0:conv_length-i-1] + sent_list[0][0:i+1]) 
            
        sent_list = header+sent_list
        
        while len(sent_list) < max_len:
            sent_list.append(dummy)
        
        currnet_label = labels_dict[ind]
        class_dummy = [currnet_label]*conv_length
        
            
        sent_list.append(class_dummy)
            #print sent_list
        doc_list.append(sent_list)
                        
    return doc_list
    #print doc_list
            
            
def merge_two(revs, tree):
    counter=0
    for i in revs:
        
        sent2 = tree[counter]
        counter += 1
        i["tree"] = sent2
        
    return revs

def get_labels2(file):
    f = open(file)
    dict = {}
    for i in f.readlines():
        index,l = i.strip().split('|')
        index = int(index)
        dict[index] = int(l)   
    return dict

def get_labels(file):
    f = open(file)
    dict = {}
    for index,i in enumerate(f.readlines()):
        #index,l = i.strip().split('|')
        #index = int(index)
        dict[index] = int(i)   
    return dict

def get_split(file):
    f = open(file)
    dict = {}
    for i in f.readlines()[1:]:
        index,l = i.strip().split(',')
        dict[int(index)] = int(l)
        
    return dict

def get_split2(size=5952):
    
    dict = {}
    for i in range(size):
        if i < 5452:
            dict[i] = 1
        else:
            dict[i] =2

    return dict

def remove_neutral(revs):
    new_revs = []
    for i in revs:
        label = i["y"]
        if label != 2:
            if label >=0 and label <= 1:
                i["y"] = 0
                i["tree"][-1] = [0]*len(i["tree"][-1])
            elif label>=3 and label <= 4:
                i["y"]  =1
                i["tree"][-1] = [1]*len(i["tree"][-1])
            new_revs.append(i)

    return new_revs

def sibling(sents, opt):
    sent_list = []
    for key in sents:
        if key == 0:
            continue  
        currnet_node = sents[key]
        word_list = []
        word_list.append(currnet_node.word)
        parent_index = currnet_node.parentindex
        parent = sents[parent_index]
        sib_list = parent.kidsindex
        if key < parent_index:
            sib_candidate = [i for i in sib_list if i < key]
            if sib_candidate == []:
                word_list.append("*START*")
            else:
                word_list.append(sents[sib_candidate.pop()].word)
            word_list.append(parent.word)
            if sib_candidate == []:
                word_list.append("*START*")
            else:
                word_list.append(sents[sib_candidate.pop()].word)
                       
        else:
            sib_candidate = [i for i in sib_list if i > key]
            if sib_candidate == []:
                word_list.append("*STOP*")
            else:
                word_list.append(sents[sib_candidate.pop(0)].word)
            word_list.append(parent.word)
            if sib_candidate == []:
                word_list.append("*STOP*")
            else:
                word_list.append(sents[sib_candidate.pop(0)].word)            
        sent_list.append(word_list[0:opt])
    return sent_list

def set_sibling(tree,labels_dict,max_len):
    
    sent_num = len(tree)
    doc_list =[]
    
    for ind,sents in enumerate(tree):
        
        
        sib_4 = sibling(sents,4)
        sent_list = sib_4
        
    
        dummy_len = len(sent_list[0])
        dummy = ["*ZERO*"]*dummy_len
        while len(sent_list) < max_len:
            sent_list.append(dummy)
            
        currnet_label = labels_dict[ind+1]
        class_dummy = [currnet_label]*dummy_len
            
            
        sent_list.append(class_dummy)        
            
        doc_list.append(sent_list)
    return doc_list

def set_sibling2(tree,labels_dict,max_len):

    sent_num = len(tree)
    doc_list =[]
    for ind,sents in enumerate(tree):
        sib_6 = sibling2(sents,6)
        sent_list = sib_6
        dummy_len = len(sent_list[0])
        dummy = ["*ZERO*"]*dummy_len
        while len(sent_list) < max_len:
            sent_list.append(dummy)

        currnet_label = labels_dict[ind]
        class_dummy = [currnet_label]*dummy_len
        sent_list.append(class_dummy)        
        doc_list.append(sent_list)
    return doc_list    

def sibling2(sents, opt):
    sent_list = []
    for key in sents:
        if key == 0:
            continue  
        currnet_node = sents[key]
        word_list = []
        word_list.append(currnet_node.word)
        
        parent_index = currnet_node.parentindex
        parent = sents[parent_index]
        word_list.append(parent.word)
        sib_list = parent.kidsindex
        if key < parent_index:
            sib_candidate = [i for i in sib_list if i < key]
            if sib_candidate == []:
                word_list.append("*START*")
            else:
                word_list.append(sents[sib_candidate.pop()].word)
            
            if sib_candidate == []:
                word_list.append("*START*")
            else:
                word_list.append(sents[sib_candidate.pop()].word)

        else:
            sib_candidate = [i for i in sib_list if i > key]
            if sib_candidate == []:
                word_list.append("*STOP*")
            else:
                word_list.append(sents[sib_candidate.pop(0)].word)
            
            if sib_candidate == []:
                word_list.append("*STOP*")
            else:
                word_list.append(sents[sib_candidate.pop(0)].word)      
        grad_parent_ind = parent.parentindex
        grad_word = sents[grad_parent_ind].word
        word_list.append(grad_word)
        sent_list.append(word_list)
    return sent_list

def get_we(sents):

    sent_list = []
    sent_len = len(sents)-1
    for key in sents:
        if key == 0:
            continue  
        currnet_node = sents[key]
        word_list = []  
        parent_index = currnet_node.parentindex
        parent_word = sents[parent_index].word
        self_word = currnet_node.word
        
        if key > parent_index:
            right = key
            left = parent_index
        else:
            right = parent_index
            left = key
            
        if left > 1:
            left_edge = sents[left-1].word
        else:
            left_edge = "*STARTWE*"
        if  right < sent_len:
            right_edge = sents[right+1].word
        else:
            right_edge = "*STOPWE*"
            
        word_list.append(left_edge)
        word_list.append(self_word)
        word_list.append(parent_word)
        word_list.append(right_edge)
        sent_list.append(word_list)
    return sent_list

def set_we(tree,labels_dict,max_len):
    sent_num = len(tree)
    doc_list =[]
    
    for ind,sents in enumerate(tree):  
        sent_list = get_we(sents)
        dummy_len = len(sent_list[0])
        dummy = ["*ZEROWE*"]*dummy_len
        while len(sent_list) < max_len:
            sent_list.append(dummy)
    
        currnet_label = labels_dict[ind]
        class_dummy = [currnet_label]*dummy_len
    
    
        sent_list.append(class_dummy)        
    
        doc_list.append(sent_list)  
        
    return doc_list

if __name__=="__main__": 
    execfile("preindex.py")
    w2v_file = "data/google_w2v.bin"   
    sent_file = "TREC/TREC_all.txt"
    tree_file = "TREC/TREC_all_tree.p"
    label_file = "TREC/label_all.txt"
    label_dict = get_labels(label_file)
    
    split_dict = get_split2(5952)    
    
    
       
       
    print "loading data...",        
    revs, vocab = build_data_cv(sent_file, split_dict, label_dict, \
                                clean_string=False)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    
    all_tree = cPickle.load(open(tree_file,"rb"))
    
    #data_sibling = set_sibling(all_tree,label_dict,max_l+8)
    data_sibling = set_sibling2(all_tree,label_dict,max_l+8)
    #data_we = set_we(all_tree,label_dict,max_l+8)
    data_tree = set_conv_sent(all_tree,label_dict,max_l+8) 
    
    new_data_tree = []
    for ind,l in enumerate(data_tree):
        new_list=[]
        for ind2,l2 in enumerate(l):
            #new_list.append(data_tree[ind][ind2]+data_sibling[ind][ind2]+data_we[ind][ind2])
            new_list.append(data_tree[ind][ind2]+data_sibling[ind][ind2])
        new_data_tree.append(new_list)
    data_tree = new_data_tree
    for i in data_tree:
        add_tree2vocab(i, vocab)
        
    
        
    revs = merge_two(revs,data_tree)
    # 2 class or 5 ?
    # revs = remove_neutral(revs)
    
    
    
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    vocab["ROOT"]=1
    vocab["*START*"]=1
    vocab["*STOP*"]=1
    vocab["*ZERO*"]=1
    #vocab["*STARTWE*"]=1
    #vocab["*STOPWE*"]=1
    #vocab["*ZEROWE*"]=1    
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
  
    cPickle.dump([revs, W, W2, word_idx_map, vocab], open("TREC/TREC_sib.p", "wb"))
    print "dataset created!"
    
