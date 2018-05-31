'''
This is an implementation of model in inference stages
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from opts import opts as options
import tensorflow as tf
from inference_wrapper import inference_wrapper
from inference_utils.gen_caption import CaptionGenerator
from vocab import vocabulary
import json
import os

def create_generate_model(opts):
    assert   isinstance(opts,options)
    with tf.variable_scope(tf.get_variable_scope(),reuse=True):
        model = inference_wrapper(opts,reuse=True)
        vocab = vocabulary(opts)
        generator = CaptionGenerator(model,vocab,beam_size=4)
        
    return generator