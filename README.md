# Improving YOLOv3
Accompanying code for Paperspace blog [Improving YOLOv3](https://blog.paperspace.com/improving-yolo/).

## Dataset Preparation
Use [Microsoft VoTT](https://github.com/Microsoft/VoTT/releases) tool to label objects.

##### Download pretrained weights
    $ cd weights/
    $ curl https://pjreddie.com/media/files/yolov3.weights --output yolov3.weights

##### Convert labelled data to YOLO format
    $ python Convert_To_YOLO_Format.py path_to_exported_files_from_the_image_tagging_step_with_VoTT
    
## Training
Use pretrained weights to finetune the YOLOv3 model using tricks mentined in [Improving YOLOv3](https://blog.paperspace.com/improving-yolo/).
```
$ train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--gradient_accumulations GRADIENT_ACCUMULATIONS]
                [--model_def MODEL_DEF] [--data_config DATA_CONFIG]
                [--pretrained_weights PRETRAINED_WEIGHTS] [--n_cpu N_CPU]
                [--img_size IMG_SIZE]
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--evaluation_interval EVALUATION_INTERVAL]
                [--compute_map COMPUTE_MAP]
                [--multiscale_training MULTISCALE_TRAINING]
```

## Credit

### [Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/abs/1902.04103)
_Zhi Zhang, Tong He, Hang Zhang, Zhongyue Zhang, Junyuan Xie, Mu Li_ <br>
### [Learning Spatial Fusion for Single-Shot Object Detection](https://arxiv.org/abs/1911.09516)
_Songtao Liu, Di Huang, Yunhong Wang_ <br>

Most of the code in this repo has been adapted from [here](https://github.com/ruinmessi/ASFF) and [here](https://github.com/eriklindernoren/PyTorch-YOLOv3).
