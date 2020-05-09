# Improving YOLOv3
Accompanying code for Paperspace blog [Improving YOLOv3](https://blog.paperspace.com/improving-yolo/).

## Dataset Preparation
Use [Microsoft VoTT](https://github.com/Microsoft/VoTT/releases) tool to label objects.

##### Download pretrained weights
    $ mkdir weights
    $ cd weights/
    $ curl https://pjreddie.com/media/files/yolov3.weights --output yolov3.weights

##### Convert labelled data to YOLO format
    $ python Convert_To_YOLO_Format.py path_to_exported_files_from_the_image_tagging_step_with_VoTT
    
##### Create yolo config file for any number of classes
    $ python custom_model.py --num_classes 6 --file_name ./config/yolo-custom-6class.cfg
    
## Training
Use pretrained weights to finetune the YOLOv3 model using tricks mentined in [Improving YOLOv3](https://blog.paperspace.com/improving-yolo/) on your data.
```
$ python -m torch.distributed.launch --nproc_per_node=1 train.py --pretrained_weights "./weights/yolov3.weights" --n_cpu 1 --ngpu 1 --distributed True --model_def ./config/yolo-custom-6class.cfg
```

## Credit

### [Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/abs/1902.04103)
_Zhi Zhang, Tong He, Hang Zhang, Zhongyue Zhang, Junyuan Xie, Mu Li_ <br>
### [Learning Spatial Fusion for Single-Shot Object Detection](https://arxiv.org/abs/1911.09516)
_Songtao Liu, Di Huang, Yunhong Wang_ <br>

Most of the code in this repo has been adapted from [here](https://github.com/ruinmessi/ASFF) and [here](https://github.com/eriklindernoren/PyTorch-YOLOv3).
