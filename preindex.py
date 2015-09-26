import cPickle
import copy

class node:

    def __init__(self, word):
        if word != None:
            self.word = word
            self.kidsword = []
            self.kidsindex = []
            self.parent = []
            self.finished = 0
            self.is_word = 1
            self.selfindex = 0
            self.parentindex = 0
            self.label = ""

            # the "ind" variable stores the look-up index of the word in the 
            # word embedding matrix We. set this value when the vocabulary is finalized
            self.ind = -1

        else:
            self.is_word = 0


if __name__=="__main__": 
    input_file  = "TREC/TREC_all_parsed.txt"
    output_file = "TREC/TREC_all_tree.p"
    #out_file = "ordered_pos.txt"
    #f = open(out_file, "w+")
    input = open(input_file)
    counter = 0
    doc = []
    for i in input.readlines():
        print i
        counter += 1
        print counter
        i = i.strip()
        i = i.strip("[]")
        #print i
        current = i.split("), ")
        node_container = {}
        ROOT = node("ROOT")
        node_container[0] = ROOT    
        
        for index,j in enumerate(current):
            label = j.split("(")[0]
            
            j = j.split("(")[1].strip(")")
            current_list = [i.strip() for i in j.split(", ")]
            #print index, current_list
            current_node = node("_".join(current_list[1].\
                                         split("-")[:-1]))
            current_node.selfindex = int(current_list[1].\
                                     split("-")[-1])
            current_node.label = label
            
            if "_".join(current_list[0].split("-")[:-1]) == "ROOT":
                current_node.parent = ROOT
            else:    
                current_node.parent = "_".join(current_list[0].\
                                               split("-")[:-1])
            current_node.parentindex = int(current_list[0].\
                                       split("-")[-1])
            node_container[current_node.selfindex] = current_node
    
        node_container1 = copy.deepcopy(node_container)
        for i in node_container1:
            current = node_container1[i]
            p_index = current.parentindex
            #print p_index
            if p_index not in node_container:
                new_node = node(current.parent)
                new_node.parent = ROOT
                new_node.selfindex = p_index
                node_container[p_index] = new_node    
        
        for i in node_container:
            current = node_container[i]
            if current.word == "ROOT":
                continue            
            p_index = current.parentindex
            #print p_index
            if p_index not in node_container:
                new_node = node(current.parent)
                new_node.parent = ROOT
                new_node.selfindex = p_index
                node_container[p_index] = new_node
                
            p = node_container[p_index]
            
            p.kidsword.append(current.word)
            p.kidsindex.append(current.selfindex)
        
        for j in node_container:
            i = node_container[j]
            print "self", i.selfindex, "parent", i.parentindex,\
                "child", i.kidsindex
        
        doc.append(node_container)
    
    #print len(doc)
    cPickle.dump(doc, open(output_file, "wb"))