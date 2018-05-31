
'''
adaptive attention for image captioning
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
from pyciderevalcap.ciderD.ciderD import CiderD
from itertools import chain

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
tf.flags.DEFINE_bool('scst',True,'whether or not perform self critical sequence training')
tf.flags.DEFINE_string('ached_tokens','data/coco-train-idxs','the file containing words information to compute CIDEr')

opts = options()
opts.data_h5 = FLAGS.data_h5
opts.data_json = FLAGS.data_json
opts.attributes_h5 = FLAGS.attributes_h5
opts.batch_size = FLAGS.batch_size
opts.learning_rate = opts.learning_rate*FLAGS.ngpu
opts.scst = FLAGS.scst

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
         
    # build the model
    for i in xrange(FLAGS.ngpu):
        with tf.device('/device:GPU:{:d}'.format(i)),tf.name_scope('model{:d}'.format(i)):
            reuse = i>0
            models.append(LanguageModel(opts,'train',reuse))
            models[i].build()
                  
    # create a function to validate
    val_fns, generators = [],[]
    with tf.device('/gpu:0'.format(i)):
        # don't use the numpy version generator, use tensorflow version generator instead
        val_fn, _ = create_val_fn(batch_size = 100)
        val_fns.append(val_fn)
        #generators.append(generator) 
    
    batch_size = FLAGS.batch_size*FLAGS.ngpu
    start_decay_steps = int(opts.nImgs//batch_size*opts.start_decay_epoches)
    decay_steps = int(opts.nImgs//batch_size*opts.decay_epoches)
    decayed_learning_rate = tf.train.exponential_decay(opts.learning_rate,
                                                       tf.maximum(models[0].step-start_decay_steps,0),
                                                       decay_steps,
                                                       opts.decay_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=decayed_learning_rate)
    
    if FLAGS.scst:
        # the difference between the reward of sampled sequence and the baseline model's sequence
        with tf.name_scope('scst_reward'):
            reward_feed = tf.placeholder(dtype=tf.float32,shape=[None])
            # The mask for the sampled sequence
            mask_feed = tf.placeholder(dtype=tf.int32,shape=[None,None])
            rewards = tf.split(reward_feed,FLAGS.ngpu,axis=0)
            masks = tf.split(mask_feed,FLAGS.ngpu,axis=0)
        loss = []
        grads = []
        # compute gradients on each gpu
        for i in xrange(FLAGS.ngpu):
            with tf.device('/device:GPU:{:d}'.format(i)),tf.name_scope('model{:d}/grads'.format(i)):
                weighted_loss = models[i].logprob*tf.to_float(masks[i])*rewards[i][:,tf.newaxis]
                # the logprob for each sentence                
                weighted_loss = tf.reduce_sum(weighted_loss)/tf.reduce_sum(tf.to_float(masks[i]))
                loss.append(weighted_loss[tf.newaxis])
                grads.append(optimizer.compute_gradients(weighted_loss))
    
    else:
        loss = []
        atten_loss = []
        words_loss = []   
        # compute gradients on each gpu
        for i in xrange(FLAGS.ngpu):
            with tf.device('/device:GPU:{:d}'.format(i)),tf.name_scope('model{:d}/grads'.format(i)):
                grads.append(optimizer.compute_gradients(models[i].batch_loss))
                loss.append(tf.expand_dims(models[i].batch_loss,axis=0))
                words_loss.append(tf.expand_dims(models[i].words_loss,axis = 0))
                atten_loss.append(tf.expand_dims(models[i].atten_loss,axis = 0))
            
    # average gradients
    average_grad = []
    with tf.name_scope('process_grad'):
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
        moving_average = ema.apply(tf.trainable_variables())
        
        # only attention loss and words loss in xe training
        if not FLAGS.scst:
            atten_loss = tf.reduce_mean(tf.concat(atten_loss,axis = 0))
            words_loss = tf.reduce_mean(tf.concat(words_loss,axis = 0))        
            
    saver = tf.train.Saver(max_to_keep = opts.max_to_keep)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    init = tf.global_variables_initializer()
    summary_ops = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.train_dir,g)
    g.finalize()

########################################################################################
########################################################################################
# TODO compute CIDEr, generate random sequence , generate baseline sequence
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
nBatches_epoch = opts.nImgs//batch_size
vocab = vocabulary(opts)

# The cider_d scorer
# TODO create the df
CiderD_Scorer = CiderD(df = FLAGS.ached_tokens)

def score_seq(gts,seqs):
    
    # check if it is valid
    assert gts.shape[0]%opts.nSeqs_per_img==0
    assert len(seqs) == 2
    assert seqs[0].shape[0] == seqs[1].shape[0] ==gts.shape[0]
    assert seqs[0].shape[0]%opts.nSeqs_per_img == 0
    
    batch_size = gts.shape[0]
    nImage = batch_size//opts.nSeqs_per_img
    gts = np.reshape(gts,[nImage,opts.nSeqs_per_img,-1])
    gts = {i: list(chain.from_iterable(convert_to_str(gts[i]))) for i in xrange(nImage)}
    
    baseline_seqs, random_seqs = seqs
    res = []
    res.extend(convert_to_str(baseline_seqs))
    res.extend(convert_to_str(random_seqs))
    res = [{'image_id':i%batch_size//opts.nSeqs_per_img ,'caption':seq} for i,seq in enumerate(res)]
    
    _, scores = CiderD_Scorer.compute_score(gts,res)
    assert len(scores) == 2*batch_size
    # baseline_seq - random_seqs
    rewards_max,rewards_random = scores[:batch_size], scores[-batch_size:]
    scores = scores[:batch_size] - scores[-batch_size:]
    #scores[:batch_size] - scores[-batch_size:]
    
    return scores, rewards_max, rewards_random

def convert_to_str(encoded_seqs):
    encoded_seqs = [[' '.join([str(w) for w in encoded_seq if w>0])] for encoded_seq in encoded_seqs]
    return encoded_seqs

def sample(sess, attributes,objects_features,image_features,freeze,max_sample):
    # sample max or randomly sample from the model
    '''A Tensorflow version of random sampling and max sampling'''
    feed_dict = {}
    fetches = []
    
    for gpu in xrange(FLAGS.ngpu):
        feed = {'model{:d}/attributes:0'.format(gpu):attributes[gpu],
                'model{:d}/objects_features:0'.format(gpu):objects_features[gpu],
                'model{:d}/image_features:0'.format(gpu):image_features[gpu],
                'model{:d}/freeze:0'.format(gpu):freeze,
                'model{:d}/max_sample:0'.format(gpu):max_sample}
        feed_dict.update(feed)
        
        fetches.append(models[gpu].seqs)
        
    seqs_np = sess.run(fetches, feed_dict)
    seqs_np = np.concatenate(seqs_np,axis = 0)
    return seqs_np

    
for i in xrange(start,FLAGS.max_iterations):
    
    attributes, features, image_features, input_seqs,target_seqs,image_ids = dataloader.get_batch(batch_size)
    
    attributes = np.split(attributes,FLAGS.ngpu,axis=0)
    features = np.split(features,FLAGS.ngpu,axis=0)
    image_features = np.split(image_features,FLAGS.ngpu,axis=0)
    input_seqs = np.split(input_seqs,FLAGS.ngpu,axis=0)
    target_seqs = np.split(target_seqs,FLAGS.ngpu,axis=0)
    freeze = i< FLAGS.freeze_step
    
    if FLAGS.scst:
        # generate baseline
        baseline_seqs = sample(sess,attributes,features,image_features,freeze,max_sample=True)
        # generate random sequence
        random_seqs = sample(sess,attributes,features,image_features,freeze,max_sample=False)
        
        # TODO compute the CIDEr difference of ground truth with baseline sequence and random sequence respectively
        # append END_TOKEN to baseline_seqs and random_seqs
        b = baseline_seqs.shape[0]
        baseline_seqs_end = np.copy(np.concatenate([baseline_seqs[:,1:],np.zeros([b,1]).astype(np.int32)],axis = 1))
        random_seqs_end = np.copy(np.concatenate([random_seqs[:,1:],np.zeros([b,1]).astype(np.int32)],axis = 1))
        cols2 = np.argmax(random_seqs_end==0,axis=1).tolist()
        cols1 = np.argmax(baseline_seqs_end==0,axis = 1).tolist()
        rows = range(0,b)
        # replace the first zero token with end token
        baseline_seqs_end[rows,cols1] = opts.END_TOKEN
        random_seqs_end[rows,cols2] = opts.END_TOKEN
        seqs = [baseline_seqs_end,random_seqs_end]
        gts = np.concatenate(target_seqs,axis = 0)
        scores, rewards_max, rewards_random = np.array(score_seq(gts,seqs))
        
        # in scst the input and target sequence are all random sequence
        input_seqs = random_seqs.astype(np.int32)
        # replace the first zero token with end_token each row
        target_seqs = random_seqs[:,1:]
        cols = np.argmax(target_seqs==0,axis=1).tolist()
        rows = range(0,target_seqs.shape[0])
        target_seqs[rows,cols] = opts.END_TOKEN
        # keep the size of input_seqs and target_seqs to be the same
        target_seqs = np.concatenate([target_seqs,np.zeros([target_seqs.shape[0],1])],axis = 1).astype(np.int32)
        mask = (target_seqs>0).astype(np.int32) 
        
        # split it for multi-gpu training
        input_seqs = np.split(input_seqs,FLAGS.ngpu,axis=0)
        target_seqs = np.split(target_seqs,FLAGS.ngpu,axis=0)        

        tf.logging.info('baseline reward {:f}'.format(np.mean(rewards_max)))
        tf.logging.info('generated reward {:f}'.format(np.mean(rewards_random)))        

    feed_dict = {}
    for gpu in xrange(FLAGS.ngpu):
        feed = {'model{:d}/attributes:0'.format(gpu):attributes[gpu],
                'model{:d}/objects_features:0'.format(gpu):features[gpu],
                'model{:d}/image_features:0'.format(gpu):image_features[gpu],
                'model{:d}/input_feed:0'.format(gpu):input_seqs[gpu],
                'model{:d}/target_feed:0'.format(gpu):target_seqs[gpu],
                'model{:d}/freeze:0'.format(gpu):freeze}
        
        if FLAGS.scst:
            feed[mask_feed] = mask
            feed[reward_feed] = scores
    
        feed_dict.update(feed)

    if freeze:
        tf.logging.info('we are freezing second layer')
    else:
        tf.logging.info('we relax second layer')    
    
    nepoches = i//nBatches_epoch            
    
    if FLAGS.scst:
        # scst
        # evaluation performance, write summary
        if(FLAGS.lang_eval and i>=FLAGS.evaluation_after and i%(FLAGS.nepoches_per_eval*nBatches_epoch)==0):
            # performance on validation
            CIDEr = val_fn(sess,beam_search = False)
            # the model is improvinig so save it
            if i>FLAGS.save_after_iterations and CIDEr>best_res:
                saver.save(sess,os.path.join(FLAGS.train_dir,'model-ckpt'),global_step=models[0].step)
                best_res = CIDEr            
            history_cider.append((CIDEr,i))
            print('history_cider')
            print(history_cider)
            
            loss, summary_str,_ = sess.run([train_step,summary_ops,moving_average],feed_dict)
            tf.logging.info('epoch:{:d},step: {:d},overall loss is {:f}'.format(nepoches,i,loss))
        
        else:
            loss,lr, _ = sess.run([train_step,decayed_learning_rate,moving_average],feed_dict)
            tf.logging.info('epoch:{:d},step: {:d},overall loss is {:f}'.format(nepoches,i,loss))
            tf.logging.info('lr{:f}'.format(lr))

            summary_diff = tf.Summary(value=[tf.Summary.Value(tag='reward_diff',simple_value=np.mean(scores))])
            summary_max = tf.Summary(value=[tf.Summary.Value(tag='reward_max',simple_value=np.mean(rewards_max))])
            summary_random = tf.Summary(value=[tf.Summary.Value(tag='reward_random',simple_value=np.mean(rewards_random))])

            writer.add_summary(summary_diff,i)
            writer.add_summary(summary_max,i)
            writer.add_summary(summary_random,i)
        
    else:
        # cross entropy training
        # evaluation performance, write summary
        if(FLAGS.lang_eval and i>=FLAGS.evaluation_after and i%(FLAGS.nepoches_per_eval*nBatches_epoch)==0):
            # performance on validation
            CIDEr = val_fn(sess)
            # the model is improvinig so save it
            if i>FLAGS.save_after_iterations and CIDEr>best_res:
                saver.save(sess,os.path.join(FLAGS.train_dir,'model-ckpt'),global_step=models[0].step)
                best_res = CIDEr            
            history_cider.append((CIDEr,i))
            print('history_cider')
            print(history_cider)
            loss,summary_str,loss1,loss2,_ = sess.run([train_step,summary_ops,atten_loss,words_loss,moving_average],feed_dict)
            writer.add_summary(summary_str,i)
            tf.logging.info('epoch:{:d},step: {:d},overall loss is {:f},attention loss is {:f},words loss is {:f}'.format(nepoches,i,loss,loss1,loss2))
                   
        else:
        
            loss,loss1,loss2,weights,attr_atten,top20_ix,_ = sess.run([train_step,atten_loss,words_loss,models[0].weights,models[0].attr_atten,models[0].top20_ix,moving_average],feed_dict)
            tf.logging.info('epoch:{:d},step: {:d},loss is {:f},attention loss is {:f},words loss is {:f}'.format(nepoches,i,loss,loss1,loss2))
            
    

        
    
