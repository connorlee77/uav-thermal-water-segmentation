# Online Self-Supervised Thermal River Segmentation


<p float="left">
<img src="https://user-images.githubusercontent.com/6981697/219913056-6a8d9ec0-4545-49e0-a88e-3b451f90c7fc.gif" height=250px float=left>
<img src="https://user-images.githubusercontent.com/6981697/219938622-dda7f8b5-9f07-4ade-9347-b51ced380884.gif" height=250px float=left>
</p>
This repository contains code necessary to run our online self-supervised river segmentation algorithm onboard a low-flying UAV. The ROS nodes were written for and tested on an Nvidia Jetson AGX Orin. It also contains all datasets used for training and validation in our paper.

## Getting started
This package subscribes to the following topics:
- `/boson/thermal/image_raw`: This provides a stream of 16 bit thermal imagery.
- `/imu/imu`: Provides the `qw, qx, qy, qz` orientation of the UAV. 

If running on custom UAV platform, please modify:
- camera settings (intrinsics, extrinsics, distortion), which are set in [`startup/config/flir_boson.py`](startup/config/flir_boson.py)
- IMU->Camera transformation, which is hardcoded in [`onr/startup/sync_thermal_imu.py`](onr/startup/sync_thermal_imu.py) and formatted as `[x, y, z, qw, qx, qy, qz]` 

## Environment setup

### Setup on Nvidia Jetson AGX Orin
1. Create your `conda` environment with Python 3.8 (or whichever Python version that is compatible with both `ros` and `pytorch`).
2. Install `ros-noetic` from source or via `robostack`. You'll likely have to install from source due to python version incompatabilities between the Jetson's PyTorch and RoboStack's `ros-noetic`. 
3. Install `pytorch` using a compatible wheel from [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) or build from source (not recommended). 
4. Install `onnxruntime` from source.
5. Install `torchvision` from source.  
6. Install the other requirements via `mamba/conda/pip` using `environment.yaml` or `requirements.txt`.
7. Install ROS python packages as needed via `mamba/conda/pip`.

### Setup on any other computer
1. Install `ros-noetic` via `robostack` or source. I recommend `robostack`. 
2. Install all requirements via `mamba/conda/pip`. No building from source required here. 

## Running on the online self-supervised algorithm
![flowchart](https://user-images.githubusercontent.com/6981697/223039939-d699cb30-0671-4349-8136-90543a037f7f.png)


1. Start the ROS master node.
```
roscore
```

2. Start the segmentation node. This takes a few seconds to initialize (loading pretrained weights, warming up network for training, etc...) so do it first. See command line arguments in [`segmentation/online_segmentation.py`](segmentation/online_segmentation.py) for more details on modifying training configurations.
```
python segmentation/online_segmentation.py --use-texture --postprocess --adapt
```

3. Start the pseudolabeling node (texture cue in example, but use motion cue if desired).
```
python pseudolabel/texture_cue.py
```

4. Start the node to preprocess data and perform horizon line estimation.
```
python startup/sync_thermal_imu.py

# or, if camera oriented upside down on UAV:
python startup/sync_thermal_imu.py --rotate-180

# Start the sky segmentation node if no IMU is available for horizon line estimation.
python sky_segmentation/segmentation.py --weights-path weights/fast_scnn.onnx
```

5. Play the rosbag file
```
rosbag play path/to/bagfile.bag
```

<p align=center>
<img src="https://user-images.githubusercontent.com/6981697/223040935-404cc49a-9789-4a14-a7f4-ae2a26532642.png" height=300px>
</p>

## Datasets

### Network pretraining datasets
The segmentation network used in this work was pretrained on RGB images from [COCO-Stuff](https://github.com/nightrome/cocostuff), [ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/), [Fuentes river dataset](https://zenodo.org/record/1003085#.Y_HqPnbMK3A) that contained water-related pixels. We supplemented with images scraped from Flickr via water- and aerial-related search terms and labeled them using an ADE20k-pretrained `ResNet50dilated + PPM_deepsup` convolutional neural network from the [`semantic-segmentation-pytorch`](https://github.com/CSAILVision/semantic-segmentation-pytorch) library. The set of scraped Flickr images with their annotations are made available below. 

| Dataset | Num. Train | Num. Validation | Water-related indices | Link |
|:---:|:---:|:---:|:---:|:---:|
|COCO-Stuff| 10977 | 458 | 148, 155, 178, 179 | [Link to download](https://github.com/nightrome/cocostuff) |
| ADE20K| 1743 | 163 | 22, 27, 61, 114, 129  | [Link to download](https://groups.csail.mit.edu/vision/datasets/ADE20K/) |
| Fuentes River | 300 | 0 | --- | [Link to download](https://zenodo.org/record/1003085#.Y_HqPnbMK3A) |
| Flickr | 1220 | 0 | --- | [TODO]() |

### Sky segmentation network training datasets
The sky segmentation network was trained on publicly-available thermal images containing sky pixels. The datasets are listed below:

| Dataset | Link |
| :---: | :---: |
| KAIST Pedestrian Segmentation| [Link to download](https://github.com/yeong5366/MS-UDA) |
| SODA | [Link to download](http://chenglongli.cn/code-dataset/) |
| MassMIND | [Link to download](https://github.com/uml-marine-robotics/MassMIND) |
| FLIR | [TODO]() |

### Annotated river/coastal scenes
The set of annotated images used for online segmentation validation contains N annotated images pulled from the Kentucky River, KY; Colorado River, CA; Castaic Lake, CA; and Duck Pier, NC sequences. 
![dataset](https://user-images.githubusercontent.com/6981697/223042570-b368398c-9e5b-420c-92c3-9628550206d7.png)

Link: TODO

### ROS bagfiles
| Location | Sequence Name | Link |
| :---: | :---: | :--- |
| Kentucky River | flight 2-1 | TODO |
| Colorado River | flight 1 | TODO |
| Colorado River | flight 3 | TODO |
| Castaic Lake | flight 2 | TODO |
| Castaic Lake | flight 4 | TODO |

## Nvidia Jetson Tips/Bugs
- Place ROS python imports after everything else has been imported. It seems to have conflicts with `skimage` specifically on the Jetson. 
