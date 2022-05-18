# Xmg
小卖柜

迁移paddle-class代码到小卖柜项目

## 配置

1. 将数据集link到dataset下
    ```
    ln -s [your dataset path] dataset
    ```

2. 将模型Link到models下
    ```
    ln -s [your mode path] models
    ```

3. 安装依赖
    ```
    pip install -r requirements.txt
    ```

4. 按需修改`configs`中配置

## usage
1. 预测
    ```
    python xmg/rec_img.py
    ```

2. 建立索引
    ```
    python python/build_gallery.py -c configs/build_xmg.yaml 
    ```

3. 训练

    参考[自定义特征提取](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/image_recognition_pipeline/feature_extraction.md)