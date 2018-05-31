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
#from scipy.misc import imshow

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('data_h5','./data/data.h5','the place of hdf5 file that contains image information')
tf.flags.DEFINE_string('data_json','./data/data.json','the place of json file that contains image meta information and vocabulary information')
tf.flags.DEFINE_string('attributes_h5','./data/tag_feats.h5','the place of h5 file that contains image attributes information')
tf.flags.DEFINE_string('train_dir','./model','the directory for training the model')
tf.flags.DEFINE_string('caption_file','result/caption_','the file which save results')
tf.flags.DEFINE_string('id','test','the id to specified')
tf.flags.DEFINE_integer('batch_size',100,'The batch_size')
tf.flags.DEFINE_bool('beam_search',False,'whether or not use beam search')

batch_size = FLAGS.batch_size
beam_search = FLAGS.beam_search
assert batch_size
opts = options()
opts.batch_size = FLAGS.batch_size
opts.data_h5 = FLAGS.data_h5
opts.data_json = FLAGS.data_json
opts.attributes_h5 = FLAGS.attributes_h5
opts.train_dir = FLAGS.train_dir


dataloader = DataLoader(opts,'test')
model = inference_wrapper(opts)

vocab = vocabulary(opts)
generator = CaptionGenerator(model,vocab,beam_size=4)
nImage = len(dataloader.test_ix)

sess = tf.Session()
checkpoint = tf.train.latest_checkpoint(opts.train_dir)
ema = tf.train.ExponentialMovingAverage(decay=0.999)
variables_to_restore = ema.variables_to_restore()
saver = tf.train.Saver(variables_to_restore,max_to_keep = 5)
saver.restore(sess,checkpoint)
print('restoring from checkpoint {:s}'.format(checkpoint))

nImage = len(dataloader.test_ix)
assert nImage%opts.batch_size==0
nBatches = nImage//opts.batch_size
res = []
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
        print('processing image:{:d}/{:d}'.format(i*batch_size+j,nImage))


current_dir = os.getcwd()
caption_file_full_path = os.path.join(current_dir,FLAGS.caption_file+FLAGS.id+'.json')  
with open(caption_file_full_path,'w') as f:
    json.dump(res,f)
    
#evaluation.myeval(caption_file_full_path)
os.system('./eval.sh {:s}'.format(caption_file_full_path))

result_file = caption_file_full_path+'.json_out.json'
with open(result_file) as f:
    result = json.load(f)
    
for metric, val in result.iteritems():
    print('{:s}:{:f}'.format(metric,val))
