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
import numpy as np

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('data_h5','./data/data.h5','the place of hdf5 file that contains image information')
tf.flags.DEFINE_string('data_json','./data/data.json','the place of json file that contains image meta information and vocabulary information')
tf.flags.DEFINE_string('attributes_h5','./data/tag_feats.h5','the place of h5 file that contains image attributes information')
tf.flags.DEFINE_string('train_dir','./model','the directory for training the model')
tf.flags.DEFINE_string('caption_file','result/caption.json','the file which save results')

opts = options()
opts.batch_size = 1
opts.data_h5 = FLAGS.data_h5
opts.data_json = FLAGS.data_json
opts.attributes_h5 = FLAGS.attributes_h5
opts.train_dir = FLAGS.train_dir


dataloader = DataLoader(opts,'test')
model = inference_wrapper(opts)

saver = tf.train.Saver()
vocab = vocabulary(opts)
generator = CaptionGenerator(model,vocab,beam_size=4)
nImage = len(dataloader.test_ix)

sess = tf.Session()
checkpoint = tf.train.latest_checkpoint(opts.train_dir)
saver.restore(sess,checkpoint)
print('restoring from checkpoint {:s}'.format(checkpoint))
res = []
attributes_tf = tf.placeholder(shape = [1,None],dtype = tf.float32)
_,top20_ix_tf = tf.nn.top_k(attributes_tf,20)
nattribute_vis = 10
for i in xrange(nImage):
    
    attributes,features, image_feature, image_ids,img = dataloader.get_batch(1)    
    caption = generator.beam_search(sess,image_feature,attributes,features)
    sentence = caption[0].sentence
    top20_weights = np.concatenate([weight[np.newaxis,:] for weight in caption[0].top20_weights],axis = 0)
    sentence = [vocab.to_word(w) for w in sentence]
    sentence = ' '.join(sentence)
    print(sentence)
    top20_ix = sess.run(top20_ix_tf,{attributes_tf:attributes})
    top20_attributes = top20_ix + 1
    #top20_attributes = ' '.join([dataloader.ix_to_attr.get(str(w),'UNK') for w in top20_attributes])
    # at each time step list top 3 attributes and its probability
    max_cols = np.argsort(top20_weights,axis = 1)
    rows = np.array([0]*top20_weights.shape[0])[:,np.newaxis]
    max_cols_top3 = max_cols[:,-nattribute_vis:][:,::-1]
    top20_attributes = top20_attributes[rows,max_cols_top3]
    # decode the attributes
    top20_attributes = top20_attributes.tolist()
    top20_attributes = [[dataloader.ix_to_attr.get(str(w),'UNK') for w in row ]for row in top20_attributes]
    print(top20_attributes)
    print(top20_weights[np.array(xrange(top20_weights.shape[0]))[:,np.newaxis],max_cols_top3])
    
    if i>20:
        raise Exception('stop')
    res.append({'image_id':image_ids[0],'caption':sentence})
    print('processing image:{:d}/{:d},image_id:{:s}'.format(i,nImage,str(image_ids[0])))
    

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
