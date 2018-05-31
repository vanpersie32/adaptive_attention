import json
class vocabulary(object):
     def __init__(self,opt):
          assert opt.data_json
          with open(opt.data_json) as f:
               json_out = json.load(f)
          self.ix_to_word = json_out['ix_to_word']
          self.start_id = len(self.ix_to_word)+1
          self.end_id = len(self.ix_to_word)+1
          
     def to_word(self,w):
          if str(w) in self.ix_to_word:
               return self.ix_to_word[str(w)]
          else:
               return ''