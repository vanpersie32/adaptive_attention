from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from opts import opts as options
import tensorflow as tf
from data_loader import DataLoader
from inference_wrapper import inference_wrapper
from inference_utils.gen_caption import CaptionGenerator
from vocab import vocabulary
import json
import os
from collections import namedtuple 
import time

def create_val_fn():
    
    FLAGS = namedtuple('FLAGS',['data_h5','data_json','attributes_h5','train_dir','caption_file'])
    FLAGS.data_h5 = './data/data.h5'
    FLAGS.data_json = './data/data.json'
    FLAGS.attributes_h5 = './data/tag_feats.h5'
    FLAGS.train_dir = './model'
    FLAGS.caption_file = 'result/caption.json'
    
    opts = options()
    opts.batch_size = 1
    opts.data_h5 = FLAGS.data_h5
    opts.data_json = FLAGS.data_json
    opts.attributes_h5 = FLAGS.attributes_h5
    opts.train_dir = FLAGS.train_dir
    
    
    dataloader = DataLoader(opts,'val')
    # reuse variables    
    with tf.variable_scope(tf.get_variable_scope(),reuse=True):
        model = inference_wrapper(opts,reuse=True)
        vocab = vocabulary(opts)
        generator = CaptionGenerator(model,vocab,beam_size=4)
        
    nImage = len(dataloader.val_ix)
    
    def val(sess):
        res = []
        t1 = 0    
        for i in xrange(nImage):
                          
            attributes,features, image_feature, image_ids,img = dataloader.get_batch(1)    
            caption = generator.beam_search(sess,image_feature,attributes,features)
            caption = caption[0].sentence
            caption = [vocab.to_word(w) for w in caption]
            caption = ' '.join(caption)
            res.append({'image_id':image_ids[0],'caption':caption})
            print('processing image:{:d}/{:d},time {:f}s'.format(i,nImage,time.time()-t1))
            t1 = time.time()    
        
        current_dir = os.getcwd()
        caption_file_full_path = os.path.join(current_dir,FLAGS.caption_file)  
        with open(caption_file_full_path,'w') as f:
            json.dump(res,f)
            
        #evaluation.myeval(caption_file_full_path)
        os.system('./eval.sh {:s}'.format(caption_file_full_path))
        
        result_file = caption_file_full_path+'.json_out.json'
        with open(result_file) as f:
            result = json.load(f)
            
        for metric, val in result.iteritems():
            print('{:s}:{:f}'.format(metric,val))
        
        # return the cider metric
        return result['CIDEr']
            
    return val
