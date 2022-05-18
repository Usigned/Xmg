
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from xmg.predictor import RecPredictor, VecSearcher, XmgPredictor
from xmg.configs import rec_config, gallery_path, image_dir
from utils.get_image_list import get_image_list, read_images
import numpy as np
from tqdm import tqdm


class Tester:
    def __init__(self, xmg_predictor=None):
        if xmg_predictor is None:
            self.xmg_predictor = Tester.load_default_model()
        else:
            self.xmg_predictor = xmg_predictor
        
        self.image_dir = image_dir
    
    @staticmethod
    def load_default_model(score_method=VecSearcher.score_argmax, return_k=5):
        rec_predictor = RecPredictor(rec_config)
        vec_searcher = VecSearcher(gallery_path, score_method, return_k=return_k)
        return XmgPredictor(rec_predictor, vec_searcher)

    def test_one_cls(self, cls_idx):
        image_list = get_image_list(os.path.join(image_dir, str(cls_idx)))
        images = read_images(image_list)
        
        labels = np.array([cls_idx] * len(image_list))
        return self.xmg_predictor.test(images, labels)
    
    def test_all_cls(self, begin=1, end=12):
        result = {}
        for cls in tqdm(range(begin, end+1)):
            result[cls] = self.test_one_cls(cls)
        return result
            

if __name__ == "__main__":
    
    import paddle
    # 实验室的paddle没装好，用不了gpu
    paddle.set_device('cpu')

    tester = Tester(Tester.load_default_model(VecSearcher.score_vote, return_k=3))
    result = tester.test_all_cls(9, 12)
    
    for k, v in result.items():
        print(k, v)