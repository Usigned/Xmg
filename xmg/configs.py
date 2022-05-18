import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from utils import config

rec_config_path = './configs/xmg_test.yaml'
image_dir = "./dataset/all"
gallery_path = './dataset/index'

rec_config = config.get_config(rec_config_path, show=False)