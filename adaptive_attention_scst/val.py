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
from generate_model import create_generate_model

def create_val_fn(batch_size):
    
    FLAGS = namedtuple('FLAGS',['data_h5','data_json','attributes_h5','train_dir','caption_file'])
    FLAGS.data_h5 = './data/data.h5'
    FLAGS.data_json = './data/data.json'
    FLAGS.attributes_h5 = './data/tag_feats.h5'
    FLAGS.train_dir = './model'
    FLAGS.caption_file = 'result/caption.json'
    
    opts = options()
    opts.batch_size = batch_size
    opts.data_h5 = FLAGS.data_h5
    opts.data_json = FLAGS.data_json
    opts.attributes_h5 = FLAGS.attributes_h5
    opts.train_dir = FLAGS.train_dir
    dataloader = DataLoader(opts,'val')       
    nImage = len(dataloader.val_ix)
    vocab = vocabulary(opts)
    generator = create_generate_model(opts)
    def val(sess,beam_search = True):
        res = []
        t1 = 0
        assert nImage%opts.batch_size==0
        nBatches = nImage//opts.batch_size
        for i in xrange(nBatches):
                          
            attributes,features, image_feature, image_ids,img = dataloader.get_batch(opts.batch_size)
            if beam_search:
                caption = generator.beam_search(sess,image_feature,attributes,features)
                caption = caption[0].sentence
                caption = [vocab.to_word(w) for w in caption]
                caption = [' '.join(caption)]
            else:
                caption = generator.sample(sess,image_feature,attributes,features,sample_max = True)
                caption = [' '.join([vocab.to_word(str(w)) for w in caption[j] if w>0]) for j in xrange(caption.shape[0])]
            for j in xrange(opts.batch_size):                
                res.append({'image_id':image_ids[j],'caption':caption[j]})
                print('processing image:{:d}/{:d},time {:f}s'.format(i*batch_size+j,nImage,time.time()-t1))
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

    # return the caption generator and the validation function        
    return val,generator
