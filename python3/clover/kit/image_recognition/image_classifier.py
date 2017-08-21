import os
import json
import cv2
import numpy as np
from functools import lru_cache
from . import _util

WEIGHT_FILENAME = 'weight.hdf5'
DATA_FILENAME   = 'data.json'

class ImageClassifier:

    def __init__(self, model_module, model_path):
        weight_path = os.path.join(model_path, WEIGHT_FILENAME)
        data_path   = os.path.join(model_path, DATA_FILENAME)
        with open(data_path,'r') as fin:
            self.data = json.load(fin)
        self.model = model_module.create_model(len(self.data['label_list']))
        self.model.load_weights(weight_path)

    def predict(self, img):
        import time
        tt = time.time()
        img_list = classifier_board_animal_model.preprocess_img(img)
        p_list_list = self.model.predict(img_list)
        score_list = np.max(p_list_list,axis=1)
        label_idx_list = np.argmax(p_list_list,axis=1)
        label_name_list = [self.data['label_list'][label_idx] for label_idx in label_idx_list]
        return label_name_list, score_list

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state classifier')
    parser.add_argument('module_name',help='module_name')
    parser.add_argument('img_file', help="img_file")
    args = parser.parse_args()
    
    module = __import__(args.module_name)
    img = _util.load_img(args.img_file)
    model_path = os.path.join('model',args.module_name)

    sc = ImageClassifier(module, model_path)

    label_list, score_list = sc.predict(img)
    assert(len(label_list)==len(score_list))
    for i in range(len(label_list)):
        print('{} {} {}'.format(i,label_list[i],score_list[i]))
