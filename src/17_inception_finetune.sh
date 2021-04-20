#!/bin/sh

# 不改变inception参数，仅训练bottlenet顶部的DNN
# 1. 先下载retrain.py: wget https://raw.githubusercontent.com/tensorflow/tensorflow/r1.4/tensorflow/examples/image_retraining/retrain.py
# 2. 执行下面程序进行训练
# 3. 参考: https://www.bilibili.com/video/BV1kW411W7pZ?p=29


python retrain.py \
    --bottleneck_dir bottleneck \
    --how_many_training_steps 200 \
    --model_dir inception_model/ \
    --output_graph output_graph.pb \
    --image_dir data/train/ \
    --output_labels output_labels.txt
