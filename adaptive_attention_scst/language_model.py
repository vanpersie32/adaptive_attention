from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
from opts import opts
from tensorflow.contrib import slim

########################################################################
class LanguageModel(object):
    """
    build language model based on objects attention and image attributes
    """

    #----------------------------------------------------------------------
    def __init__(self,opt,phase,reuse = False):
        """Constructor"""
        assert isinstance(opt,opts)
        self.opt = opt
        # attributes information
        self.attri = None
        # feature information of each object in the image
        self.objects_features = None
        # input_mask: weight for each word in input_seqs
        self.input_mask = None
        # input seqs for language model
        self.input_seqs = None
        # target seqs for language model
        self.target_seqs = None
        # whole image features
        self.image_features = None
        
        # train/validation/inference
        self.phase = phase
        self.batch_loss = None
        # language mode's batch_size is image batch_size * nSeqs_per_img
        # it is because each image has nSeqs_per_img labels
        self.batch_size = self.opt.batch_size * self.opt.nSeqs_per_img
        # global time step
        self.step = None
        self.length = None
        # whether or not reuse variables
        self.reuse = reuse
        # atten_loss initialize 0
        self.atten_loss = 0.0
        # weight for atten_loss
        self.lamb = 0.2
        self.logprob = []

    def build_inputs(self):
        '''
        three sources of inputs: objects features, image attributes, previous words
        '''
        if self.phase == 'train':
            
            attri = tf.placeholder(dtype = tf.float32,shape = [self.batch_size,
                                                               self.opt.attr_size],
                                                      name='attributes')
        
            objects_features = tf.placeholder(dtype=tf.float32,
                                              shape=[self.batch_size,
                                                     self.opt.nRegions,
                                                     self.opt.image_encoding_size],name='objects_features')
        
            image_features = tf.placeholder(dtype=tf.float32,
                                            shape=[self.batch_size,
                                                   self.opt.image_encoding_size],name='image_features')
            
            input_seqs = tf.placeholder(dtype=tf.int32,shape=[self.batch_size,None],name='input_feed')
            target_seqs = tf.placeholder(dtype=tf.int32,shape = [self.batch_size,None],name = 'target_feed')
            self.input_seqs = input_seqs
            self.target_seqs = target_seqs
            # check size 
            # if the number of words in input_seqs equals the number of words in target_seqs
            with tf.control_dependencies([tf.assert_equal(tf.cast(tf.not_equal(input_seqs,0),tf.int32),
                                                                    tf.cast(tf.not_equal(target_seqs,0),tf.int32))]):
                self.input_mask = tf.cast(tf.not_equal(input_seqs,0),tf.int32)
            
            self.attri = attri
            self.objects_features = objects_features
            self.image_features = image_features
            self.freeze = tf.placeholder(shape = [],dtype = tf.bool,name = 'freeze')
            
        else:
            
            # At inference step: one image per batch
            attri = tf.placeholder(dtype = tf.float32,shape = [None,
                                                               self.opt.attr_size],
                                                      name='attributes')
        
            objects_features = tf.placeholder(dtype=tf.float32,
                                              shape=[None,
                                                     self.opt.nRegions,
                                                     self.opt.image_encoding_size],name='objects_features')
        
            image_features = tf.placeholder(dtype=tf.float32,
                                            shape=[None,
                                                   self.opt.image_encoding_size],name='image_features')
            
            # feed all previous words(history information)
            # the sequence length is unknown, the batch size is unknown
            input_seqs = tf.placeholder(dtype=tf.int32,shape=[None,None],name = 'input_feed')
                        
            batch_size = tf.shape(input_seqs)[0]
            
            self.input_seqs = input_seqs
            self.target_seqs = None
            self.input_mask = None
            
            self.attri = attri
            self.objects_features = objects_features
            self.image_features = image_features
    
    def seq_embedding(self,word,reuse = False):
        with tf.variable_scope('seq_embedding',reuse = reuse):
            emb_matrix = tf.get_variable('map',
                                         shape=[self.opt.vocab_size,
                                                self.opt.input_encoding_size],
                                         initializer=tf.contrib.layers.xavier_initializer(),
                                         dtype= tf.float32)

            # the attributes embeddings is from row 1~999
            # attributes embedding share weights with words embeddings
            # 0 is a null token
            ix = tf.constant(range(1,1+self.opt.attr_size))
            self.attr_matrix = tf.gather(emb_matrix,ix)

            word_embedding = tf.nn.embedding_lookup(emb_matrix,word)            
            return word_embedding
        
    def build_forward_step(self):
        
        '''This is a step of forwarding'''
        # build the model in reuse mode or in unreuse mode
        # in train: unreuse, validation: reuse(reuse variable in train mode), test: unreuse
            
        # first level attention
        # attributes attention, previous words attention, objects attention

        def first_layer_attention(step,h,all_atten,all_inputs,reuse):
        
            def attributes_attention(step,attributes,h,attri_atten,reuse = None):
                # attention on each attribute
                with tf.variable_scope('attributes_att',reuse = reuse) as scope:
                    mapped_size = 512
                    attr_size = self.opt.attr_size
                    # it is a matrix that record the feature of each attributes
                    # share weights with word embeddings
                    attr_matrix = self.attr_matrix
                    
                    h_emb = slim.fully_connected(h,
                                                 mapped_size,
                                                 biases_initializer=None,
                                                 activation_fn=None,
                                                 reuse = reuse,
                                                 scope='h_emb')
                    
                    # select top 20 attributes
                    # DONT put it in the loop
                    #top20_prob,top20_ix = tf.nn.top_k(attributes,20)
                    top20_prob,top20_ix = self.top20_prob,self.top20_ix 
                    top20_emb = tf.gather(attr_matrix,top20_ix)
                    # mapping top20 attributes and h to the same space
                    top20_map = slim.fully_connected(tf.nn.relu(top20_emb),
                                                     mapped_size,
                                                     biases_initializer = None,
                                                     activation_fn = None,
                                                     reuse = reuse,
                                                     scope = 'top20_map')
                    score = slim.fully_connected(tf.reshape(tf.nn.tanh(h_emb[:,tf.newaxis,:]+top20_map),[-1,mapped_size]),
                                                 1,
                                                 biases_initializer = None,
                                                 activation_fn = None,
                                                 reuse = reuse,
                                                 scope = 'score')

                    weights = tf.nn.softmax(tf.reshape(score,[-1,20]))
                    assert(isinstance(attri_atten,tf.TensorArray))
                    if self.phase == 'train':
                        mask_t = tf.to_float(self.input_mask[:,step])[:,tf.newaxis]
                    else:
                        mask_t = 1.0
                    
                    new_attri_atten = attri_atten.write(step,weights*mask_t)
                    
                    # weights* probability * embedding
                    weighted_emb = weights[:,:,tf.newaxis]*top20_emb*top20_prob[:,:,tf.newaxis]
                    context = tf.reduce_sum(weighted_emb,axis = 1)
                    if self.phase == 'train':
                        # compute attention correctness
                        # attributes index in the vocabulary
                        eps = 1e-7
                        top20_attributes = top20_ix +1 
                        target_seq = self.target_seqs[:,step]
                        mask = tf.equal(top20_attributes,target_seq[:,tf.newaxis])
                        atten_loss = -tf.log(tf.boolean_mask(weights,mask)+eps)
                        atten_loss = tf.reduce_sum(atten_loss)
                        self.atten_loss = self.atten_loss + atten_loss
                                    
                return context, new_attri_atten
            
            def objects_attention(step,objects_features,h,obj_atten,reuse):
                
                # attention on each objects
                with tf.variable_scope('objects_att',reuse = reuse) as scope:
                    mapped_size = 512
                    obj_emb = slim.conv2d(objects_features,
                                          mapped_size,
                                          kernel_size=[1],
                                          activation_fn=None,
                                          biases_initializer=None,
                                          reuse = reuse,
                                          scope = 'obj_emb')
                    
                    nRegions = tf.shape(obj_emb)[1]
                    
                    h_emb = slim.fully_connected(h,
                                                 mapped_size,
                                                 activation_fn=None,
                                                 biases_initializer=None,
                                                 reuse = reuse,
                                                 scope = 'h_emb')
                    
                    score = slim.fully_connected(tf.reshape(tf.nn.tanh(obj_emb + tf.expand_dims(h_emb,axis=1)),
                                                            [-1,mapped_size]),
                                                 1,
                                                 activation_fn=None,
                                                 biases_initializer=None,
                                                 reuse = reuse,
                                                 scope = 'score')
                    
                    score = tf.reshape(score,[-1,nRegions])
                    
                    weights = tf.nn.softmax(score)
                    
                    context = tf.reduce_sum(tf.expand_dims(weights,axis=2)*objects_features,axis=1)
                    
                    assert(isinstance(obj_atten,tf.TensorArray))
                    
                    new_obj_atten = obj_atten.write(step,weights)
                    
                    return context, new_obj_atten
                    
            ###########################################################################
                
            
            
            # attention on attributes, objects feature, history(previously generated words)
            
            with tf.variable_scope('first_att',reuse = reuse):
                attributes, objects_features, word_embeddings = all_inputs
                attrib_atten, obj_atten = all_atten
                # use attributes attention
                attri_context, new_attri_atten = attributes_attention(step,attributes,h,attrib_atten,reuse = reuse)
                # don't use attributs attention, directly use attributes information
                #new_attri_atten = tf.TensorArray(dtype = tf.float32,size = 10)
                #attri_context = tf.identity(attributes)
                objects_context, new_obj_atten = objects_attention(step,objects_features,h,obj_atten,reuse = reuse)
                #history_context, new_history_atten = history_attention(step,h,word_embeddings,history_atten,reuse)                  
                
                all_outputs = [attri_context,objects_context]
                all_new_att = [new_attri_atten,new_obj_atten]
            
            return all_outputs, all_new_att
            

        # second layer attention
        def second_layer_attention(step,attri_context,obj_context,h,second_atten,reuse):
            with tf.variable_scope('second_att',reuse = reuse):
                mapped_size = 512
                
                attri_linear = slim.fully_connected(attri_context,
                                                    mapped_size,
                                                    activation_fn=tf.nn.relu,
                                                    scope='attri_linear',
                                                    reuse = reuse)
                
                attri_emb = slim.fully_connected(attri_linear,
                                                 mapped_size,
                                                 activation_fn=None,
                                                 scope='attr_emb',
                                                 reuse = reuse)
                
                obj_linear = slim.fully_connected(obj_context,
                                                  mapped_size,
                                                  activation_fn = tf.nn.relu,
                                                  scope = 'obj_linear',
                                                  reuse = reuse)
                
                obj_emb = slim.fully_connected(obj_linear,
                                               mapped_size,
                                               activation_fn=None,
                                               scope='obj_emb',
                                               reuse = reuse)
 
                h_emb = slim.fully_connected(h,
                                             mapped_size,
                                             activation_fn=None,
                                             scope='h_emb',
                                             reuse = reuse)
                
                inputs = tf.concat([tf.expand_dims(attri_emb,axis=1),
                                     tf.expand_dims(obj_emb,axis=1)],axis=1)
                
                score = slim.fully_connected(tf.reshape(tf.nn.tanh(tf.expand_dims(h_emb,axis = 1)+inputs),[-1,mapped_size]),
                                             1,
                                             activation_fn=None,
                                             biases_initializer=None,
                                             reuse = reuse,
                                             scope='score')
                
                score = tf.reshape(score,[-1,2])
                weights = tf.nn.softmax(score)
                if self.phase == 'train':
                    weights = tf.cond(self.freeze,lambda:tf.constant([[0.5,0.5]]),lambda: weights)
                
                context = weights[:,0::2]*attri_linear
                context = context + weights[:,1::2]*obj_linear
                #context = context + weights[:,2::3]*history_linear
                
                assert isinstance(second_atten,tf.TensorArray)
                if self.phase == 'train':
                    mask_t = self.input_mask[:,step][:,tf.newaxis]
                else:
                    mask_t = tf.constant(1.0)
                new_second_atten = second_atten.write(step,weights*tf.cast(mask_t,tf.float32))
                
                return context, new_second_atten       
            
        # control wether or not should reuse parameters in attended_lstm
        def attended_lstm(step,states,loss,first_atten,second_att,all_inputs,save_logprob,reuse):
            
            attributes, objects_features, seq_embeddings = all_inputs
            # build_attention
            c,h = tf.split(states,2,axis=1)
            new_outputs,new_first_att = first_layer_attention(step,h,first_atten,all_inputs,reuse)
            context,new_second_att = second_layer_attention(step,new_outputs[0],new_outputs[1],h,second_att,reuse = reuse)

            # lstm
            def lstm_cell(inputs,(c,h)):
                '''lstm cell inplementations'''
                seq_embedding,  context = inputs
                        
                i2h = slim.fully_connected(seq_embedding,
                                           4*self.opt.rnn_size,
                                           activation_fn=None,
                                           biases_initializer=tf.contrib.layers.xavier_initializer(),
                                           reuse = reuse,
                                           scope='i2h')
                
                h2h = slim.fully_connected(h,
                                           4*self.opt.rnn_size,
                                           activation_fn=None,
                                           biases_initializer=tf.contrib.layers.xavier_initializer(),
                                           reuse = reuse,
                                           scope='h2h')
                
                context2h = slim.fully_connected(context,
                                                 4*self.opt.rnn_size,
                                                 activation_fn=None,
                                                 biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                 reuse = reuse,
                                                 scope='context2h')
                
                
                all_input_sums = i2h+h2h+context2h
                reshaped = tf.reshape(all_input_sums,[-1,4,self.opt.rnn_size])
                n1, n2, n3, n4 = reshaped[:,0],reshaped[:,1],reshaped[:,2],reshaped[:,3]
                in_gate = tf.nn.sigmoid(n1)
                forget_gate = tf.nn.sigmoid(n2)
                out_gate = tf.nn.sigmoid(n3)
                in_transform = tf.nn.tanh(n4)
                next_c = forget_gate*c+in_gate*in_transform
                next_h = out_gate*tf.nn.tanh(next_c)
                return next_h,(next_c,next_h)
            
                
            
            with tf.variable_scope('lstm',reuse = reuse):
                # three kinds of information: sentence information,attributes information, context(attention over objects)
                seq_embedding = seq_embeddings[:,step]
                inputs = [seq_embedding, context]
                lstm_output,new_states = lstm_cell(inputs,(c,h))
            new_states = tf.concat(new_states,axis=1,name='new_states')                
            
            # _outputProb   
            with tf.variable_scope('logits',reuse=reuse):
                MidSize = 1024
                all_sum = tf.add_n([
                                    slim.fully_connected(lstm_output,
                                                         MidSize,
                                                         reuse=reuse,
                                                         activation_fn=None,
                                                         biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                         scope='h_emb3'),
                                    
                                    slim.fully_connected(context,
                                                         MidSize,
                                                         reuse=reuse,
                                                         biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                         activation_fn=None,
                                                         scope='context_emb3'),
                                    slim.fully_connected(seq_embedding,
                                                         MidSize,
                                                         reuse=reuse,
                                                         biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                         activation_fn=None,
                                                         scope='seq_emb3')])
                all_sum = tf.nn.relu(all_sum)
                
                logits = slim.fully_connected(all_sum,
                                              self.opt.vocab_size,
                                              activation_fn=None,
                                              reuse = reuse,
                                              biases_initializer=tf.contrib.layers.xavier_initializer(),
                                              scope='logits2')                     

            # criterion
            if self.phase == 'train':
                with tf.name_scope('loss'):
                    word_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,
                                                                               labels=self.target_seqs[:,step])
                    if save_logprob:
                        self.logprob.append(-word_loss)
                    weight = tf.cast(self.input_mask[:,step],tf.float32)
                    new_loss = loss + tf.reduce_sum(word_loss*weight) 
                
                return step+1,new_states,new_loss,new_first_att,new_second_att,all_inputs,logits

            # calculate probability over vocabulary
            else:
                tf.nn.softmax(logits,name='prob')
                return 0, new_states, 0.0, new_first_att,new_second_att,all_inputs, logits
        
        # return the attended_lstm function
        return attended_lstm
            
                
    def build_model(self):
        '''
        build model based on objects attention and image attributes
        '''
        # return the create a step of forward passing function
        forward_fn = self.build_forward_step()
        self.forward_fn = forward_fn
        
        # for preparation of forward passing
        
        reuse = self.reuse
        
        
        # generate lstm initial state
        with tf.variable_scope('init',reuse = reuse):

            h = slim.fully_connected(self.image_features,
                                     self.opt.rnn_size,
                                     biases_initializer=tf.contrib.layers.xavier_initializer(),
                                     scope='F2H',
                                     reuse = reuse)
            
            c = slim.fully_connected(self.image_features,
                                     self.opt.rnn_size,
                                     biases_initializer=tf.contrib.layers.xavier_initializer(),
                                     scope='F2C',
                                     reuse = reuse)
             
        init_states = tf.concat([c,h],axis=-1,name='init_states')
        
        # max length of the sequence
        if self.input_mask is not None:
            length =tf.reduce_max(tf.reduce_sum(self.input_mask,axis=1))
        else:
            length = tf.shape(self.input_seqs)[1]
            
            
        # seq_embeddings BxTxfeature_dim
        seq_embeddings = self.seq_embedding(self.input_seqs,reuse = reuse)
        

        # do it by looping
        loss = tf.constant(0.0)
        states = init_states
        # the attention of first level
        first_atten = [tf.TensorArray(dtype=tf.float32,
                                     size=self.opt.seq_length+1),
                       tf.TensorArray(dtype=tf.float32,
                                      size=self.opt.seq_length+1)]
        
       
        # the attention of second level
        second_atten = tf.TensorArray(dtype=tf.float32,
                                      size=self.opt.seq_length+1)
        all_inputs = [self.attri,self.objects_features,seq_embeddings]
        
        # top 20 attributes probability and their index
        self.top20_prob, self.top20_ix = tf.nn.top_k(self.attri,20)

        # in train mode : loop
    
        if self.phase == 'train':
            for i in xrange(self.opt.seq_length+1):
                # a step forward attended lstm
                reuse = reuse or (i!=0)
                _,states, loss,first_atten,second_atten, all_inputs,_ = forward_fn(tf.constant(i),states,loss,first_atten,second_atten,all_inputs,True,reuse)
                        
            words_loss = tf.div(loss,
                                tf.cast(tf.reduce_sum(self.input_mask),tf.float32))
            self.logprob = tf.concat([logprob[:,tf.newaxis] for logprob in self.logprob],axis=1)
            
            self.words_loss = words_loss
            # how many words in input_seqs are attributes
            top20_attributes = self.top20_ix +1
            ntop20 = tf.reduce_sum(tf.cast(tf.equal(self.input_seqs[:,:,tf.newaxis],top20_attributes[:,tf.newaxis,:]),tf.float32))
            # in case ntop20 become zero
            ntop20 = tf.maximum(ntop20,1.0)
            self.atten_loss = self.atten_loss/ntop20
            # batch loss == words_loss + lambda* atten_loss
            self.batch_loss = words_loss + self.lamb* self.atten_loss
            batch_loss = self.batch_loss
            self.length = i
            self.weights = tf.transpose(second_atten.stack(),[1,0,2])
            self.attr_atten = tf.transpose(first_atten[0].stack(),[1,0,2])
            tf.summary.scalar('batch_loss',batch_loss)
            
        else:
            # in inference mode: a single forward passing
            state_feed = tf.placeholder(dtype=tf.float32,shape = [None,self.opt.rnn_size*2],name='state_feed')
            # seq_embeddings bxtxfeature_dims
            step = tf.shape(seq_embeddings)[1]-1
            all_inputs = [self.attri,self.objects_features,seq_embeddings]            
            _,new_states,_,first_atten,second_atten,_,_= forward_fn(step,state_feed,tf.constant(0,dtype = tf.float32),first_atten,second_atten,all_inputs,False,reuse)
        
            
    def build_step(self):
        
        with tf.variable_scope('global_step',reuse = self.reuse):
            self.step = tf.get_variable(name='step',
                                        shape=[],
                                        dtype=tf.int32,
                                        initializer=tf.constant_initializer(value=0,dtype=tf.int32),
                                        trainable=False)
    
    def build(self):
        # build inputs
        self.build_inputs()
        # build the model
        # weight and bias initializer
        with slim.arg_scope([slim.fully_connected],
                            biases_initializer = tf.contrib.layers.xavier_initializer(),
                            weights_initializer = tf.contrib.layers.xavier_initializer()):
            
            self.build_model()            
        # build step
        if self.phase == 'train':
            self.build_step()
            if self.opt.scst:
                self.seqs, self.seq_log_probs = self.sample()
             
    
    def sample(self):
        '''This builds sampling from the model, only used in scst'''
        
        assert self.phase=='train'
        reuse = True
        # prepare inputs
        with tf.variable_scope('init',reuse = reuse):

            h = slim.fully_connected(self.image_features,
                                     self.opt.rnn_size,
                                     biases_initializer=tf.contrib.layers.xavier_initializer(),
                                     scope='F2H',
                                     reuse = reuse)
            
            c = slim.fully_connected(self.image_features,
                                     self.opt.rnn_size,
                                     biases_initializer=tf.contrib.layers.xavier_initializer(),
                                     scope='F2C',
                                     reuse = reuse)
             
        init_states = tf.concat([c,h],axis=-1)
        states = init_states
        
        # the attention of first level
        first_atten = [tf.TensorArray(dtype=tf.float32,
                                     size=self.opt.seq_length+1),
                       tf.TensorArray(dtype=tf.float32,
                                      size=self.opt.seq_length+1)]
        
       
        # the attention of second level
        second_atten = tf.TensorArray(dtype=tf.float32,
                                      size=self.opt.seq_length+1)
        
        # one step forward function
        forward_fn = self.forward_fn
        
        batch_size = tf.shape(self.attri)[0]
        seq_log_probs = []
        seq = []
        
        def sample_fn(logits,max_sample):
            
            def sample_max():
                sample_log_prob = tf.to_float(tf.reduce_max(tf.nn.log_softmax(logits),axis=-1)[:,tf.newaxis])
                it = tf.to_int32(tf.argmax(logits,axis=-1)[:,tf.newaxis])
                return it, sample_log_prob
            def sample_random():
                it = tf.to_int32(tf.multinomial(logits,1))
                log_prob = tf.nn.log_softmax(logits)
                sample_log_prob = tf.gather(tf.transpose(log_prob,[1,0]),tf.squeeze(it,axis=1))
                sample_log_prob = tf.to_float(tf.diag_part(sample_log_prob)[:,tf.newaxis])
                return it, sample_log_prob
            
            it, sample_log_prob = tf.cond(max_sample,sample_max,sample_random)
            return it, sample_log_prob
            
        max_sample = tf.placeholder(dtype=tf.bool,shape=[],name='max_sample')
        
        for i in xrange(self.opt.seq_length+1):
            
            if i==0:
                it = tf.ones(shape=[batch_size,1],dtype=tf.int32)*self.opt.START_TOKEN
                
            else:
                
                it, sample_log_prob = sample_fn(logits,max_sample)
                                                    
            if i>0:
                if i==1:
                    unfinished = tf.cast(tf.logical_and(tf.not_equal(it,self.opt.END_TOKEN),
                                                        tf.not_equal(it, 0)),tf.int32)
                else:
                    unfinished = unfinished*tf.cast(tf.logical_and(tf.not_equal(it,self.opt.END_TOKEN),
                                                                   tf.not_equal(it,0)),tf.int32)
                # replace end_token with zero
                it = tf.cast(it,tf.int32) * unfinished
                seq.append(it)
                seq_log_probs.append(sample_log_prob*tf.cast(unfinished,tf.float32))
            
            if i==0:
                xt = it
                seq.append(it)
            else:
                xt = tf.concat(seq,axis=1,name='concat{:d}'.format(i+3))
            seq_embeddings = self.seq_embedding(tf.stop_gradient(xt),reuse = True)
            all_inputs = [self.attri,self.objects_features,seq_embeddings]                
            _, states, _, first_atten, second_atten, all_inputs,logits = forward_fn(tf.constant(i),
                                                                                    states,
                                                                                    tf.constant(0.0),
                                                                                    first_atten,
                                                                                    second_atten,
                                                                                    all_inputs,
                                                                                    save_logprob = False,
                                                                                    reuse = True)
            
            
        return tf.concat(seq,axis=1,name='concat2'), tf.concat(seq_log_probs,axis = 1,name='concat1')
                
        
        
    
    
