from opts import opts
import h5py
import json
import numpy as np
from vocab import vocabulary
import cPickle
import random
import os

class DataLoader(object):
    '''
      dataloader for loading data
    '''
    def __init__(self,opt,phase):
        assert isinstance(opt,opts)
        self.opt = opt

        # new attributes are extracted by resnet
        # store in h5 file dataset feats
        self.attributes = h5py.File(opt.attributes_h5)['feats']
        data = h5py.File(opt.data_h5)
        with open(opt.data_json) as f:
            json_out = json.load(f)
        splits = [image_info['split'] for image_info in json_out['images']]
        image_ids = [image_info['image_id'] for image_info in json_out['images']]
        
        self.image_ids = image_ids
        self.train_ix = [ i for i,split in enumerate(splits)  if split==0]
        self.val_ix = [i  for i,split in enumerate(splits) if split==1]
        self.test_ix = [i for i,split in enumerate(splits) if split==2]
        self.pt = 0
        self.phase = phase
        self.features = data['features']
        self.labels = data['labels']
        self.label_start_ix = data['label_start_ix']
        self.label_end_ix = data['label_end_ix']
        assert self.labels.shape[-1] == self.opt.seq_length
        self.ix_to_word = json_out['ix_to_word']
        self.ix_to_attr = json_out['ix_to_attr']
        # decide the value of start token and end token
        if self.opt.END_TOKEN and self.opt.START_TOKEN:
            assert self.opt.END_TOKEN == self.opt.START_TOKEN
            assert self.opt.START_TOKEN == len(json_out['ix_to_word'])+1
        else:
            self.opt.vocab_size = len(json_out['ix_to_word'])+2
            self.opt.END_TOKEN = self.opt.START_TOKEN = len(json_out['ix_to_word'])+1
        
    def get_batch(self,batch_size):
        
        if self.phase == 'train':
            data_ix = self.train_ix
        elif self.phase == 'val':
            data_ix = self.val_ix
        else:
            data_ix = self.test_ix
        
        attributes = np.zeros((batch_size,self.opt.attr_size)).astype(np.float32)
        features = np.zeros((batch_size,self.opt.nRegions,self.opt.image_encoding_size)).astype(np.float32)
        image_feature = np.zeros((batch_size,self.opt.image_encoding_size)).astype(np.float32)
        input_seqs = np.zeros((batch_size,self.opt.nSeqs_per_img,self.opt.seq_length+1)).astype(np.int32)
        target_seqs = np.zeros((batch_size,self.opt.nSeqs_per_img,self.opt.seq_length+1)).astype(np.int32)
        nRegions = self.opt.nRegions
        image_ids = []
        # first step: start token
        input_seqs[:,:,0] = self.opt.START_TOKEN
        
        for i in xrange(batch_size):
            # reset iterator if it has reached the end
            if self.pt>=len(data_ix):
                # shuffle each epoch
                random.shuffle(data_ix)
                self.pt = 0
            index = data_ix[self.pt]
            features[i] = np.reshape(self.features[index*nRegions:(index+1)*nRegions,:]
                                     ,(nRegions,-1))
            image_feature[i] = features[i][-1]
            # the index is 1-based in hdf5, but it is 0-based in python
            st = self.label_start_ix[index]
            ed = self.label_end_ix[index]
            # randomly pick five labels
            pick = np.random.randint(st,ed-self.opt.nSeqs_per_img+1)
            nseqs = self.opt.nSeqs_per_img
            input_seqs[i,:,1:] = self.labels[pick:pick+nseqs,:]
            target_seqs[i,:,0:-1] = self.labels[pick:pick+nseqs,:]
            image_ids.append(self.image_ids[index])
            attributes[i] = self.attributes[index]
            self.pt +=1

            
        # language model batch_size: batch_size * self.opt.nSeqs_per_img
        if self.phase == 'train':        
            attributes = np.repeat(attributes,self.opt.nSeqs_per_img,axis=0)
            features = np.repeat(features,self.opt.nSeqs_per_img,axis=0)
            image_feature = np.repeat(image_feature,self.opt.nSeqs_per_img,axis=0)
            input_seqs = np.reshape(input_seqs,[-1,self.opt.seq_length+1])
            target_seqs = np.reshape(target_seqs,[-1,self.opt.seq_length+1])
            # END TOKEN at last
            cols = np.argmax(np.int32(target_seqs==0),axis=1)
            rows = np.array(xrange(cols.shape[0]))
            #for row,col in enumerate(indices):
            #target_seqs[row,col] = self.opt.END_TOKEN
            target_seqs[rows,cols] = self.opt.END_TOKEN 
            #vocab = vocabulary(self.opt)
                        
            return attributes, features, image_feature, input_seqs, target_seqs, image_ids
        else:
            return attributes,features, image_feature, image_ids, None
            
    
