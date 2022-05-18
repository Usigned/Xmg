import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from python.preprocess import create_operators
from python.postprocess import build_postprocess
from utils.predictor import Predictor
from utils.get_image_list import get_image_list, read_images
from xmg.configs import rec_config, gallery_path
import numpy as np
import os
import faiss
import pickle


class RecPredictor(Predictor):
    '''
    特征提取器，将图片转换为特征向量，用于后续向量搜索
    '''
    def __init__(self, config):
        super().__init__(config["Global"],
                         config["Global"]["rec_inference_model_dir"])
        self.preprocess_ops = create_operators(config["RecPreProcess"][
            "transform_ops"])
        self.postprocess = build_postprocess(config["RecPostProcess"])
        self.benchmark = config["Global"].get("benchmark", False)
    
    def predict(self, images, feat_norm=True):
        input_names = self.predictor.get_input_names()
        input_tensor = self.predictor.get_input_handle(input_names[0])

        output_names = self.predictor.get_output_names()
        output_tensor = self.predictor.get_output_handle(output_names[0])
        
        if not isinstance(images, (list, )):
            images = [images]
        
        for idx in range(len(images)):
            for ops in self.preprocess_ops:
                images[idx] = ops(images[idx])
        image = np.array(images)
        input_tensor.copy_from_cpu(image)
        self.predictor.run()
        batch_output = output_tensor.copy_to_cpu()
        
        if feat_norm:
            feas_norm = np.sqrt(
                np.sum(np.square(batch_output), axis=1, keepdims=True))
            batch_output = np.divide(batch_output, feas_norm)
            
        
        if self.postprocess is not None:
            batch_output = self.postprocess(batch_output)
            
        return batch_output
    

class VecSearcher:
    def __init__(self, path, score_method, return_k=5):
        self.searcher, self.id_map = VecSearcher.load_gallery(path)
        self.score_method = score_method
        self.return_k = return_k    
    
    @staticmethod
    def load_gallery(path):
        idx_file = os.path.join(path, 'vector.index')
        searcher = faiss.read_index(idx_file)
        id_map_path = os.path.join(path, 'id_map.pkl')
        with open(id_map_path, 'rb') as fd:
            id_map = pickle.load(fd)
        
        # 转换后获得 idx: cls_idx的map
        # idx, cls_idx均为int类型
        id_map = {
            key: int(value.split(' ')[-1]) 
            for key, value in id_map.items()
        }
        
        return searcher, id_map


    def predict(self, rec_result, score_method=None):
        if score_method is None:
            score_method = self.score_method
            
        scores, docs = self.searcher.search(rec_result, self.return_k)
        preds = np.ndarray((rec_result.shape[0]), dtype='int')

        for i, (score, doc) in enumerate(zip(scores, docs)):
            preds[i] = score_method(score, doc, self.id_map)
        return preds
        
    @staticmethod
    def score_argmax(score, doc, id_map):
        return id_map[doc[0]]

    @staticmethod
    def score_vote(score, doc, id_map):
        result = {}
        for s, d in zip(score, doc):
            idx = id_map[d]
            result[idx] = result.get(idx, 0) + s
        return sorted(result, key=lambda id: result[id])[-1]



class XmgPredictor:
    '''
    主要包含特征提取模型rec_model和gallery searcher两部分
    1. 使用rec_model将图片提取为特征向量
    2. 使用searcher将向量与库中向量进行比对
    '''
    
    def __init__(self, rec_predictor, vec_searcher):
        self.rec_predictor = rec_predictor
        self.vec_searcher = vec_searcher

    def predict(self, images):
        feat_vecs = self.rec_predictor.predict(images)
        pred_idx = self.vec_searcher.predict(feat_vecs)
        return pred_idx

    def test(self, images, labels):
        preds = self.predict(images)
        return np.mean(np.equal(preds, labels))
    

if __name__ == "__main__":
    
    import paddle
    # 实验室的paddle没装好，用不了gpu
    paddle.set_device('cpu')

    
    rec_predictor = RecPredictor(rec_config)
    vec_searcher = VecSearcher(gallery_path, VecSearcher.score_argmax)
    
    xmg_predictor = XmgPredictor(rec_predictor, vec_searcher)
    
    
    image_list = get_image_list("./dataset/all/12")
    images = read_images(image_list)
    
    labels = np.array([12] * len(image_list))
    
    print(xmg_predictor.test(images, labels))