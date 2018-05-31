'''
training options for image captioning
'''
class opts(object):
    
    def __init__(self):
        self.data_h5 = ''
        self.data_json = ''
        # language mode's batch_size is image batch_size * nSeqs_per_img
        # it is because each image has nSeqs_per_img labels
        self.batch_size = 24
        self.nRegions = 11
        self.max_iterations = 100000
        self.learning_rate = 5e-5
        #self.learning_rate_decay = 0.5
        #self.decay_steps = 5000
        self.grad_clip = 5.0
        self.nSeqs_per_img = 5
        self.num_eval = 5000
        self.checkpoint = ''
        self.vocab_size = None
        self.input_encoding_size = 512
        self.image_encoding_size = 2054
        self.rnn_size = 512
        self.seq_length = 25
        self.num_layers = 1
        #self.dropout = 0.5
        # change attributes size from 1000 to 999, new attributes are extracted 
        # using resnet
        self.attr_size = 999
        # new attributes file, extracted using resnet
        self.START_TOKEN = None
        self.END_TOKEN = None
        self.learning_rate_decay_rate = 0.5
        self.learning_rate_decay_step = 50000
        self.max_to_keep = 40
        self.input_sequence_size = 20
        self.nImgs = 120000
        
        
