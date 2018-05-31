from language_model import LanguageModel
import tensorflow as tf

class inference_wrapper(object):
    
    def __init__(self,opt,reuse = False):
        # build the model
        # separate inference op with train op, especially in train and validation steps
        with tf.name_scope('inference'):
            LM = LanguageModel(opt,'test',reuse = reuse)
            LM.build()
        self.model = LM
    
    def inference_step(self,sess,objects_features,attributes,input_feed,state_feed):
        
        feed_dict = {'inference/objects_features:0':objects_features,
                     'inference/input_feed:0':input_feed,
                     'inference/attributes:0':attributes,
                     'inference/state_feed:0':state_feed}
        
        prob, new_state,top20_weights = sess.run(['inference/prob:0','inference/new_states:0',self.model.top20_weights],feed_dict)
        
        return prob, new_state, None, top20_weights
        
    
    def init_state(self,sess,image_features,input_feed):
        feed_dict = {'inference/image_features:0':image_features,
                     'inference/input_feed:0':input_feed}
        init_state = sess.run('inference/init_states:0',feed_dict)
        return init_state
    
