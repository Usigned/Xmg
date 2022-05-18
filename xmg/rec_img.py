import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import paddle
# 实验室的paddle没装好，用不了gpu
paddle.set_device('cpu')

from python.preprocess import create_operators
from python.postprocess import build_postprocess
from utils.predictor import Predictor
from utils.get_image_list import get_image_list
from utils import config
import numpy as np
import cv2
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
    

def read_imgs(image_list):
    '''
    从路径列表中读取图片
    '''
    imgs = []
    names = []
    
    for img_path in image_list:
        img = cv2.imread(img_path) # image (H, W, C-BGR)

        img = img[:, :, ::-1] # bgr2rgb
        imgs.append(img)
        img_name = os.path.basename(img_path)
        names.append(img_name)
    return imgs, names
    

def inference_by_batch(imgs, predictor, batch_size=32):
    '''
    按batch来预测，默认是一个一个预测
    有bug没修好，不要使用
    '''
    cnt = 0
    batch_imgs = []
    res = []
    for idx, img in enumerate(imgs):
        batch_imgs.append(img)
        cnt += 1
        if cnt % batch_size == 0 or (idx + 1) == len(imgs):
            if len(batch_imgs) == 0:
                continue
            batch_results = predictor.predict(batch_imgs)
            res.append(batch_results)
            batch_imgs = []
    
    return np.concatenate(res, axis=0)


def vote(docs, scores, id_map):
    '''
    使用投票选取最佳的结果
    '''
    res = {}
    for score, doc in zip(scores, docs):
        idx = int(id_map[doc].split(' ')[-1])
        res[idx] = res.get(idx, 0) + score
    return sorted(res, key=lambda id: res[id])[-1]


def load_gallery(index_dir):
    '''
    加载向量库
    '''
    idx_file = os.path.join(index_dir, 'vector.index')
    searcher = faiss.read_index(idx_file)
    id_map_path = os.path.join(index_dir, 'id_map.pkl')
    with open(id_map_path, 'rb') as fd:
        id_map = pickle.load(fd)
    return searcher, id_map

class XmgPredictor:
    '''
    主要包含特征提取模型rec_model和gallery searcher两部分
    1. 使用rec_model将图片提取为特征向量
    2. 使用searcher将向量与库中向量进行比对
    '''
    
    def __init__(self, rec_config_path, gallery_path, vote=False):
        conf = config.get_config(rec_config_path, show=False)
        self.rec_model = RecPredictor(conf)
        # id_map => dict
        self.searcher, self.id_map = load_gallery(gallery_path)
        self.vote = vote
    
    def predict(self, images):
        # 返回(n, 512) ndarray float32
        rec_result = self.rec_model.predict(images)
        return_k = 5
        # scores: (n, return_k)  float32
        # docs: (n, return_k) int64
        scores, docs = self.searcher.search(rec_result, return_k)
        
        predict_result = []
        for doc, score in zip(docs, scores):
            if self.vote:
                pred_idx = vote(doc, score, self.id_map)
            else:
                pred_idx = int(self.id_map[doc[0]].split(' ')[-1])
            predict_result.append(pred_idx)
        return docs, predict_result
    
    def test_class(self, cls_idx, image_list):
        '''
        测试单个类别的正确率
        cls_idx:       类别编号
        image_list:    测试图片的地址
        '''
        imgs, _ = read_imgs(image_list)
        _, predict_result = self.predict(imgs)
        
        cnt = 0
        for pred_idx in predict_result:
            if pred_idx == cls_idx:
                cnt += 1
        return cnt / len(image_list), predict_result

    def test_images(self, image_list, label_list):
        imgs, _ = read_imgs(image_list)
        docs, predict_result = self.predict(imgs)
        # print(predict_result)
        wrongs = []
        cnt = 0
        for doc, fname, pred_idx, label in zip(docs, image_list, predict_result, label_list):
            if int(pred_idx) == int(label):
                cnt += 1
            else:
                wrongs.append((fname, pred_idx, doc))
        return cnt, wrongs
    

if __name__ == "__main__":

    searcher, id_map = load_gallery("./dataset/index")
    # print(id_map)

    data_dir = './dataset/all'
    rec_config = './configs/xmg_test.yaml'
    gallery_path = './dataset/index'
    predictor = XmgPredictor(rec_config, gallery_path)
    

    precision_dict = {}
    for idx in range(12, 13):
        path = os.path.join(data_dir, str(idx))
        image_list = get_image_list(path)
        precision_dict[idx], res = predictor.test_class(idx, image_list)
    print(precision_dict)

    # image_dir = './xmg_dataset/'
    # train_list = []
    # train_label = []

    # val_list = []
    # val_label = []


    # with open(image_dir + "train_data.txt", 'r') as f:
    #     for line in f.readlines():
    #         fpath, label = line.split(' ')
    #         train_list.append(os.path.join(image_dir, fpath))
    #         train_label.append(label)

    
    # with open(image_dir + "val_data.txt", 'r') as f:
    #     for line in f.readlines():
    #         fpath, label = line.split(' ')
    #         val_list.append(os.path.join(image_dir, fpath))
    #         val_label.append(label)


    # def batch_inferences(image_list, label_list, batch_size=64):
    #     cnt, l = 0, len(image_list)
    #     wrongs = []
    #     while len(image_list) != 0:
    #         batch, image_list = image_list[:64], image_list[64:]
    #         batch_label, label_list = label_list[:64], label_list[64:]
    #         tmp, _wrongs = predictor.test_images(batch, batch_label)
    #         cnt += tmp
    #         wrongs += _wrongs
    #     return cnt / l, wrongs

    # prec, wrongs = batch_inferences(train_list, train_label)
    # print(prec)
    # with open("wrongs.txt", 'w') as f:
    #     for wrong in wrongs:
    #         f.write(f"{wrong[0]} {id_map[wrong[2][0]]}\n")