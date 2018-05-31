
'''
dual level attention for image captioning
first level: Attention on objects, Attention on attributes, Attention on previously generated words
second level: Attention on the output of the first level(objects context, attributes context, history context)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from opts import opts as options
from language_model import LanguageModel
import tensorflow as tf
from data_loader import DataLoader
import os
from val import create_val_fn
import numpy as np
from vocab import vocabulary

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size',16,'batch size for training')
tf.flags.DEFINE_string('data_h5','./data/data.h5','the place of hdf5 file that contains image information')
tf.flags.DEFINE_string('data_json','./data/data.json','the place of json file that contains image meta information and vocabulary information')
tf.flags.DEFINE_string('attributes_h5','./data/tag_feats.h5','the place of h5 file that contains image attributes information')
tf.flags.DEFINE_string('train_dir','./model','the directory for training the model')
tf.flags.DEFINE_integer('max_iterations',1000000,'the maximam iteration of training the model')
tf.flags.DEFINE_float('clip_value',5.0,'clip gradients to prevent scalar exceed clip_value')
tf.flags.DEFINE_float('save_after_iterations',15000,'how many iterations before we save our checkpoints')
tf.flags.DEFINE_integer('ngpu',2,'the number of gpu to use')
tf.flags.DEFINE_integer('freeze_step',20000,'how many step we freeze second attention')
tf.flags.DEFINE_integer('nepoches_per_eval',1,'how many epoches before we evaluate the model')
tf.flags.DEFINE_integer('evaluation_after',25000,'how many iterations before we evaluate our methods')
tf.flags.DEFINE_bool('lang_eval',True,'whether or not evaluate the language model')

opts = options()
opts.data_h5 = FLAGS.data_h5
opts.data_json = FLAGS.data_json
opts.attributes_h5 = FLAGS.attributes_h5
opts.batch_size = FLAGS.batch_size
opts.learning_rate = opts.learning_rate*FLAGS.ngpu
dataloader = DataLoader(opts,'train')
g = tf.Graph()

tf.logging.set_verbosity(tf.logging.INFO)

def gradients_clipping(grads_params):
    new_grads_params = []
    for g,p in grads_params:
        clipped_g = tf.clip_by_value(g,-FLAGS.clip_value,FLAGS.clip_value)
        new_grads_params.append((clipped_g,p))
    return new_grads_params



models = []
grads = []
with g.as_default():
    
    optimizer = tf.train.AdamOptimizer(learning_rate=opts.learning_rate)
    loss = []
    atten_loss = []
    words_loss = []   
    # compute gradients on each gpu
    for i in xrange(FLAGS.ngpu):
        with tf.device('/device:GPU:{:d}'.format(i)),tf.name_scope('model{:d}'.format(i)):
            reuse = i>0
            models.append(LanguageModel(opts,'train',reuse))
            models[i].build()
            grads.append(optimizer.compute_gradients(models[i].batch_loss))
            loss.append(tf.expand_dims(models[i].batch_loss,axis=0))
            words_loss.append(tf.expand_dims(models[i].words_loss,axis = 0))
            atten_loss.append(tf.expand_dims(models[i].atten_loss,axis = 0))
            
    # average gradients
    average_grad = []        
    for grad_and_var in zip(*grads):
        grad = []
        for gra,_ in grad_and_var:
            grad.append(tf.expand_dims(gra,axis=0))
            
        grad = tf.reduce_mean(tf.concat(grad,axis=0),0)
        average_grad.append((grad,grad_and_var[0][1]))
        
    clipped_gradients = gradients_clipping(average_grad)
    
    # apply gradients
    train_op = optimizer.apply_gradients(clipped_gradients,models[0].step)

    # create exponential moving average
    ema = tf.train.ExponentialMovingAverage(decay=0.999,num_updates = models[0].step)
    
    with tf.control_dependencies([train_op]):
        # moving average update parameters, return overall loss, attention loss, words_loss
        train_step = tf.reduce_mean(tf.concat(loss,axis=0))
        atten_loss = tf.reduce_mean(tf.concat(atten_loss,axis = 0))
        words_loss = tf.reduce_mean(tf.concat(words_loss,axis = 0))
        moving_average = ema.apply(tf.trainable_variables())
            
    saver = tf.train.Saver(max_to_keep = opts.max_to_keep)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    init = tf.global_variables_initializer()
    summary_opts = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.train_dir,g)
    # create a function to validate
    with tf.device('/gpu:0'):
        val_fn = create_val_fn()
    g.finalize()
    

checkpoint = tf.train.latest_checkpoint(FLAGS.train_dir)
if checkpoint:
    saver.restore(sess,checkpoint)
    tf.logging.info('restoring from checkpoint')
else:
    sess.run(init)
    
start = sess.run(models[0].step)

best_res = 0.0
history_cider = []
batch_size = FLAGS.batch_size*FLAGS.ngpu
# the number of batches(iterations) per epoch
nBatches_epoch = int(opts.nImgs/batch_size)

vocab = vocabulary(opts)

for i in xrange(start,FLAGS.max_iterations):
    
    attributes, features, image_feature, input_seqs,target_seqs,image_ids = dataloader.get_batch(batch_size)
    attributes = np.split(attributes,FLAGS.ngpu,axis=0)
    features = np.split(features,FLAGS.ngpu,axis=0)
    image_feature = np.split(image_feature,FLAGS.ngpu,axis=0)
    input_seqs = np.split(input_seqs,FLAGS.ngpu,axis=0)
    target_seqs = np.split(target_seqs,FLAGS.ngpu,axis=0)
    feed_dict = {}
    freeze = i< FLAGS.freeze_step
    for gpu in xrange(FLAGS.ngpu):
        feed = {'model{:d}/attributes:0'.format(gpu):attributes[gpu],
                'model{:d}/objects_features:0'.format(gpu):features[gpu],
                'model{:d}/image_features:0'.format(gpu):image_feature[gpu],
                'model{:d}/input_feed:0'.format(gpu):input_seqs[gpu],
                'model{:d}/target_feed:0'.format(gpu):target_seqs[gpu],
                'model{:d}/freeze:0'.format(gpu):freeze}
    
        feed_dict.update(feed)

    if freeze:
        tf.logging.info('we are freezing second layer')
    else:
        tf.logging.info('we relax second layer')         
    
    nepoches = int(i/nBatches_epoch)            
    # evaluate performance per epoch
    if(FLAGS.lang_eval and i>=FLAGS.evaluation_after and i!=0 and i%(FLAGS.nepoches_per_eval*nBatches_epoch)==0):
        # perform validation
        CIDEr = val_fn(sess)
        # the model is improvinig so save it
        if i>FLAGS.save_after_iterations and CIDEr>best_res:
            saver.save(sess,os.path.join(FLAGS.train_dir,'model-ckpt'),global_step=models[0].step)
            best_res = CIDEr
        
        loss,summary_str,loss1,loss2,_ = sess.run([train_step,summary_opts,atten_loss,words_loss,moving_average],feed_dict)
        writer.add_summary(summary_str,i)
        history_cider.append((CIDEr,i))
        print('history_cider')
        print(history_cider)
        tf.logging.info('epoch:{:d},step: {:d},overall loss is {:f},attention loss is {:f},words loss is {:f}'.format(nepoches,i,loss,loss1,loss2))
    
    else:
        loss,loss1,loss2,weights,attr_atten,top20_ix,_ = sess.run([train_step,atten_loss,words_loss,models[0].weights,models[0].attr_atten,models[0].top20_ix,moving_average],feed_dict)
        tf.logging.info('epoch:{:d},step: {:d},loss is {:f},attention loss is {:f},words loss is {:f}'.format(nepoches,i,loss,loss1,loss2))
