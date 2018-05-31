# adaptive attention: combing object attention and attribute attention for image captioning
This is the implementation of adaptive attention. Besides, we add semantic attention correctness and exponential moving average when training our model. As fixed attention model is a special case of adaptive attention model, we can also train the it by setting freeze_step in the train.py to a very large value. The whole model can be trained within a day and CIDEr should reaches to 1.092 on MS COCO test set with EMA and 1.072 without EMA.

## How to use
1. train the model with cross entropy loss
2. train the model with self critical sequence learning

### Train the adaptiive attention model with cross entropy loss
Type the command line and train adaptive attention with cross entropy loss:
```bash
cd ~/adaptive_attention_XELoss
train.py --freeze_step -1 2>&1 | tee res.log
```
You can monitor the history performance on validation set by reading the res.log and we only save the best model on validation set. The performance of our model will peak after 20 epochs.

### Train the fixed attention model with cross entropy loss
As fixed attention is a special case of adaptive attention, the code can also be used for training fixed attention with the following command line:
```bash
cd ~/adaptive_attention_XELoss
train.py --freeze_step 100000000 2>&1 |tee res.log
```
You can fixed the model by setting freeze_step to a very large value in command line. We can also change the relative weight to attribute attention and object attention in language_model.py
by changing the following line 
```bash
weights = tf.cond(self.freeze,lambda:tf.constant([[0.5,0.5]]),lambda: weights. 
```
We set weights of attribute attention and object attention to 0.5 and 0.5 respectively by default.

### Train the adaptive attention model with self critical sequence learning
```bash
cp ~/adaptive_attention_XELoss/model/* ~/adaptive_attention_scst/model/
cd ~/adaptive_attention_scst/model
train.py --freeze_step -1 2>&1 |tee res.log
```
After 20 epochs, the performance(CIDEr) of adaptive attention model will reach to 1.172 on MS COCO test set and 1.13 on MS COCO test server
