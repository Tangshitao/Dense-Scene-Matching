This repository contains code of our CVPR 2021 paper - ["Learning Camera Localization via Dense Scene Matching"]() by Shitao Tang, Chengzhou Tang, Rui Huang, Siyu Zhu and Ping Tan.

This paper presents a new method for scene agnostic camera localization using dense scene matching (DSM), where a cost volume is constructed between a query image and a scene. The cost volume and the corresponding coordinates are processed by a CNN to predict dense coordinates. Camera poses can then be solved by PnP algorithms.

If you find this project useful, please cite:
```
@inproceedings{Tang2021Learning,
  title={Learning Camera Localization via Dense Scene Matching},
  author={Shitao Tang, Chengzhou Tang, Rui Huang, Siyu Zhu and Ping Tan},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```

## Usage
### Environment
* The codes are tested along with 
  - pytorch=1.4.0
  - lmdb (optional)
  - yaml
  - skimage
  - opencv
  - numpy=1.17
  - tensorboard
### Installation
* Build PyTorch operations
  ```
    cd libs/model/ops
    python setup.py install
  ```
* Build PnP algorithm
  ```
    cd libs/utils/lm_pnp
    mkdir build
    cd build
    cmake ..
    make all
  ```
### Train and Test
* Download

  You can download the trained models and label files for [7scenes](), [Cambridge](), [Scannet]().

  For 7scenes, you can use the prepared data in the following.

  |[Chess]() |[Fire]() |[Heads]() |[Office]() |[Pumpkin]() |[Kitchen]() |[Stairs]() |
  |:-:|:-:|:-:|:-:|:-:|:-:|:-:|

  For Cambridge landmarks, you can download image files [here](http://mi.eng.cam.ac.uk/projects/relocalisation/), and depths [here](https://heidata.uni-heidelberg.de/api/access/datafile/:persistentId?persistentId=doi:10.11588/data/EGCMUU/7LBIQJ)

* Test
  
  Please refer to configs/7scenes.yaml for detailed explaination of how to set label file path and image file path 
  * 7scenes
    ```
    python tools/video_test.py --config configs/7scenes.yaml
    ```
  * Camrbrige
    ```
    python tools/video_test.py --config configs/cambridge.yaml
    ```

* Train

  We use ResNet-FPN pretrained [model]()
  ```
    python tools/train_net.py
  ```