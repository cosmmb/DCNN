import cPickle
import time
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
import copy
warnings.filterwarnings("ignore")   

#different non-linearities
def ReLU(x):
    y = T.maximum(0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)
       
def train_conv_net(datasets,
                   U,
                   img_w=300, 
                   filter_hs=[3,4,5],
                   hidden_units=[100,2], 
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=20, 
                   batch_size=50, 
                   lr_decay = 0.95,
                   conv_non_linear="relu",
                   use_valid_set=True,
                   activations=[Iden],
                   sqr_norm_lim=9,
                   non_static=True):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes    
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """    

    rng = np.random.RandomState(3435)
    # data[0] is training and data[1] is testing
    img_h = len(datasets[0][0])-1  
    # m length of word embedding
    filter_w = img_w    
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    # prepare the feature maps here
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, 1, filter_w*filter_h))
        pool_sizes.append((img_h-1+1, img_w-filter_w+1))

    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("learn_decay",lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static)
                    ,("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch)]
    print parameters    
    
    #define model architecture
    index = T.lscalar()

    x = T.tensor3('x',dtype='float32') 
    y = T.ivector('y')
    U = np.array(U, dtype='float32')
    Words = theano.shared(value = U, name = "Words")
    zero_vec_tensor = T.vector(dtype='float32')
    zero_vec = np.zeros(img_w,dtype='float32')
    #### initialization
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))],allow_input_downcast=True)
    
    ########################### tree CNN start
    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_hs)):
        layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]*x.shape[-1]))[:,:,:,0:img_w*filter_hs[i]]
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w*filter_hs[i]),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)    
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    ########################### tree CNN end
    
    
    
    ########################### sibling start
    sib = [range(1500,1800),range(1800,2100),range(2100,2400),range(2400,2700),range(2700,3000)]
    sib_templet = [sib[0]+sib[2],sib[0]+sib[1]+sib[2],sib[0]+sib[2]+sib[3],sib[0]+sib[1]+sib[2]+sib[3],sib[0]+sib[1]+sib[2]+sib[4]]    
    filter_shapes_sib = []
    pool_sizes_sib = []    
    filter_sib = [2,3,3,4,4]
    for filter_h in filter_sib:
        filter_shapes_sib.append((feature_maps, 1, 1, filter_w*filter_h))
        pool_sizes_sib.append((img_h-1+1, img_w-filter_w+1))    
        
    for i in xrange(len(filter_sib)):
        layer0_input_sib = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]*x.shape[-1]))[:,:,:,sib_templet[i]]
        filter_shape_sib = filter_shapes_sib[i]
        pool_size_sib = pool_sizes_sib[i]
        conv_layer_sib = LeNetConvPoolLayer(rng, input=layer0_input_sib,image_shape=(batch_size, 1, img_h, img_w*filter_sib[i]),
                                        filter_shape=filter_shape_sib, poolsize=pool_size_sib, non_linear=conv_non_linear)        
        layer1_input_sib = conv_layer_sib.output.flatten(2)
        conv_layers.append(conv_layer_sib)
        layer1_inputs.append(layer1_input_sib)    
        
    ########################### sibling end
        

    layer1_input = T.concatenate(layer1_inputs,1)
    hidden_units[0] = feature_maps*(len(filter_hs)+len(filter_sib)) ##m.m.   
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
    
    #define parameters of the model and update functions using adadelta
    params = classifier.params     
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]
    cost = classifier.negative_log_likelihood(y) 
    dropout_cost = classifier.dropout_negative_log_likelihood(y)           
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)
    
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate 
    #extra data (at random)
    np.random.seed(3435)
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = np.random.permutation(datasets[0])   
        extra_data = train_set[:extra_data_num]
        new_data=np.append(datasets[0],extra_data,axis=0)
    else:
        new_data = datasets[0]
    new_data = np.random.permutation(new_data)
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches*0.9))
    if len(datasets)==3:
        print "3 splits!"
        use_valid_set=True
        train_set = new_data
        val_set = datasets[1]
        train_set_x, train_set_y = shared_dataset((train_set[:,:img_h],train_set[:,-1,0]))
        val_set_x, val_set_y = shared_dataset((val_set[:,:img_h],val_set[:,-1,0]))
        test_set_x = datasets[2][:,:img_h] 
        test_set_y = np.asarray(datasets[2][:,-1,0],"int32")
        n_val_batches = int(val_set.shape[0] / batch_size)
        val_model = theano.function([index], classifier.errors(y),
            givens={
                  x: val_set_x[index * batch_size: (index + 1) * batch_size],
                  y: val_set_y[index * batch_size: (index + 1) * batch_size]},allow_input_downcast=True)
    else:
        test_set_x = datasets[1][:,:img_h] 
        test_set_y = np.asarray(datasets[1][:,-1,0],"int32")
        
        if use_valid_set:
            train_set = new_data[:n_train_batches*batch_size,:]
            val_set = new_data[n_train_batches*batch_size:,:]     
            train_set_x, train_set_y = shared_dataset((train_set[:,:img_h],train_set[:,-1,0]))
            val_set_x, val_set_y = shared_dataset((val_set[:,:img_h],val_set[:,-1,0]))
            n_val_batches = n_batches - n_train_batches
            val_model = theano.function([index], classifier.errors(y),
                 givens={
                    x: val_set_x[index * batch_size: (index + 1) * batch_size],
                     y: val_set_y[index * batch_size: (index + 1) * batch_size]},allow_input_downcast=True)
        else:
            train_set = new_data[:,:]    
            train_set_x, train_set_y = shared_dataset((train_set[:,:img_h],train_set[:,-1]))  
            
    # make theano functions to get train/val/test errors
    test_model = theano.function([index], classifier.errors(y),
             givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                 y: train_set_y[index * batch_size: (index + 1) * batch_size]},allow_input_downcast=True)               
    train_model = theano.function([index], cost, updates=grad_updates,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
              y: train_set_y[index*batch_size:(index+1)*batch_size]},allow_input_downcast=True)     
    test_pred_layers = []
    test_split_num = 10
    test_size = test_set_x.shape[0]/test_split_num
    
    
    
    for i, conv_layer in enumerate(conv_layers):
        if i < len(filter_hs):
            test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]*x.shape[-1]))[:,:,:,0:img_w*filter_hs[i]]
        else:
            test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]*x.shape[-1]))[:,:,:,sib_templet[i-len(filter_hs)]] 
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
        
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_y_pred = classifier.predict(test_layer1_input)
    test_error = T.mean(T.neq(test_y_pred, y))
    test_model_all = theano.function([x,y], test_error,allow_input_downcast=True)   
    
    #start training over mini-batches
    print '... training'
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0       
    cost_epoch = 0
    best_test = 0
    while (epoch < n_epochs):     
        t0 = time.time()
        epoch = epoch + 1
        if shuffle_batch:
            counter = 0
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                counter += 1
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)  
                set_zero(zero_vec)
        train_losses = [test_model(i) for i in xrange(n_train_batches)]
        train_perf = 1 - np.mean(train_losses)
        val_losses = [val_model(i) for i in xrange(n_val_batches)]
        val_perf = 1- np.mean(val_losses)
        
        test_result_list = []
        for i in range(test_split_num):
            temp_error = test_model_all(test_set_x[i*test_size:(i+1)*test_size],test_set_y[i*test_size:(i+1)*test_size])
            test_result_list.append(temp_error)
        
        test_perf = 1- float(np.mean(test_result_list))
        
        t1 = time.time()
        used_time = (t1-t0)/60
        if val_perf >= best_val_perf:
            best_val_perf = val_perf
            best_test = test_perf  
        
        print('epoch %i, train perf %f %%, val perf %f, test perf %f, time %f' % (epoch, train_perf * 100., val_perf*100., test_perf*100., used_time))
        f.write('epoch %i, train perf %f %%, val perf %f, test perf %f, time %f' % (epoch, train_perf * 100., val_perf*100., test_perf*100., used_time))
        f.write("\n")
    return best_test

def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
        
def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)
    
def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to
    
def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
        else:
            print word
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def get_idx_for_tensor (sent, word_idx_map, max_l, k, filter_h):
    each_sent = copy.deepcopy(sent)
    for j, each_word in enumerate(each_sent[:-1]):
        for l, each_field in enumerate(each_word):
            if each_field in word_idx_map:
                each_sent[j][l] = word_idx_map[each_field]
            elif each_field == 0:
                continue
            else:
                print each_field
                
    return each_sent

def merge_sent_and_tree(sent, tree):
    conv_width = 5
    l = []
    tree_size = len(tree)
    new_tree = []
    for i in range(len(sent)-conv_width):
        l.append(sent[i:i+conv_width])
        
    while len(l) < tree_size:
        l.append([0]*conv_width)
        
    for idx, i in enumerate(tree):
        new_tree.append(i + l[idx])
    new_tree[-1] = [tree[-1][0]]*len(new_tree[-1])
    return new_tree
    

def make_idx_data(revs, word_idx_map, cv, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test, dev = [], [], []
    train_tensor, test_tensor, dev_tensor =[], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)   
        sent.append(rev["y"])
        sent_tensor = get_idx_for_tensor(rev["tree"], word_idx_map, max_l, k, filter_h)
        #remove the following line to take seq cnn into consideration
        #sent_tensor = merge_sent_and_tree(sent, sent_tensor)
        if rev["split"]==2:            
            test.append(sent)
            test_tensor.append(sent_tensor)
        elif rev["split"]==1:
            train.append(sent)
            train_tensor.append(sent_tensor)
        elif rev["split"]==3:
            dev.append(sent)
            dev_tensor.append(sent_tensor)        
    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    dev = np.array(dev,dtype="int")
    train_tensor = np.array(train_tensor,dtype="int")
    test_tensor = np.array(test_tensor,dtype="int")  
    dev_tensor = np.array(dev_tensor,dtype="int")
    return [train,test], [train_tensor,test_tensor]      
  
   
if __name__=="__main__":
    print "loading data...",
    x = cPickle.load(open("TREC/TREC_sib.p","rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    word_idx_map["ROOT"] = 0
    # revs = ["y"","text,"split","num_words"]
    print "data loaded!"
    input = sys.argv[1]
    mode = "-nonstatic"
    word_vectors = "-word2vec"
    #word_vectors = sys.argv[2]
    f = open("log_"+ input +".txt",'w')
    if mode=="-nonstatic":
        print "model architecture: CNN-non-static"
        non_static=True
    elif mode=="-static":
        print "model architecture: CNN-static"
        non_static=False
    execfile("conv_net_classes.py")    
    if word_vectors=="-rand":
        print "using: random vectors"
        U = W2
    elif word_vectors=="-word2vec":
        print "using: word2vec vectors"
        U = W
    results = []
    r = range(0,10)    
    for i in [0]:
        datasets, datasets_tensor = make_idx_data(revs, word_idx_map, i, max_l=56,k=300, filter_h=5)
        perf = train_conv_net(datasets_tensor,
                              U,
                              lr_decay=0.95,
                              filter_hs=[3,4,5],
                              conv_non_linear="relu",
                              hidden_units=[100,6], 
                              use_valid_set=True, 
                              shuffle_batch=True, 
                              n_epochs=20, 
                              sqr_norm_lim=9,
                              non_static=non_static,
                              batch_size=int(input),
                              dropout_rate=[0.5])
        print "cv: " + str(i) + ", perf: " + str(perf)
        f.write("cv: " + str(i) + ", perf: " + str(perf))
        f.write("\n")
        results.append(perf)
  
    print str(np.mean(results))
    f.write(str(np.mean(results)))
    f.close()
