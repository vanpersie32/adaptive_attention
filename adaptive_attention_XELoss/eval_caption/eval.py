'''
 evaluate image captionining with bleu,meator,cider metric
'''
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
import json
from json import encoder
import sys

def myeval(caption_file):

    encoder.FLOAT_REPR = lambda o: format(o, '.3f')
    
    annotate_file = './annotations/captions_val2014.json'
    result_file = '%s.json_out.json' %(caption_file)
    coco = COCO(annotate_file)
    print(caption_file)
    cocoRes = coco.loadRes(caption_file)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
    
    result = cocoEval.eval
    with open(result_file,'w') as f:
        json.dump(result,f)

if __name__ == '__main__':
    myeval(sys.argv[1])


