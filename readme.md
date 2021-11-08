This repository contains code of our CVPR 2021 paper - ["Learning Camera Localization via Dense Scene Matching"](https://arxiv.org/abs/2103.16792) by Shitao Tang, Chengzhou Tang, Rui Huang, Siyu Zhu and Ping Tan.

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

  You can download the trained models and label files for [7scenes](https://drive.google.com/file/d/174bx6EvKWM4BT6JCsh-uwaJrlRn04eL5/view?usp=sharing), [Cambridge](https://drive.google.com/file/d/1vshSXWt6dPG10Qg68yzhmpI_JLWcsjpk/view?usp=sharing), [Scannet](https://drive.google.com/file/d/1iiFFMy-8MpUjvj3tPCgW6XXQMyeUx6Uq/view?usp=sharing).

  For 7scenes, you can use the prepared data in the following.

  |[Chess](https://drive.google.com/file/d/1BBARBi5CO-0h-JTUPxGZldzvnNeVBwHq/view?usp=sharing) |[Fire](https://drive.google.com/file/d/1-ToNOrrL3IacRjpuTb8V1xRMGCRVzC-j/view?usp=sharing) |[Heads](https://drive.google.com/file/d/1hELSgft4-NmZ7AiBOktDiVia4JaEI_0I/view?usp=sharing) |[Office](https://drive.google.com/file/d/1Y1W3UyO6Sr4lD57yV410d_AekeDb3SKo/view?usp=sharing) |[Pumpkin](https://drive.google.com/file/d/1m3g0wexIHmVJ0-02-cpr9BFN8ikyNthM/view?usp=sharing) |[Kitchen](https://drive.google.com/file/d/1Fn5pQEb64HZXdCaV33TlXGKJYm3GfdRw/view?usp=sharing) |[Stairs](https://drive.google.com/file/d/1haZ1B-SHrqY9MV-KrpzzjeQvtPw-QxZJ/view?usp=sharing) |
  |:-:|:-:|:-:|:-:|:-:|:-:|:-:|

  For Cambridge landmarks, you can download image files [here](http://mi.eng.cam.ac.uk/projects/relocalisation/), and depths [here](https://heidata.uni-heidelberg.de/api/access/datafile/:persistentId?persistentId=doi:10.11588/data/EGCMUU/7LBIQJ).

* Test
  
  Please refer to configs/7scenes.yaml for detailed explaination of how to set label file path and image file path.
  * 7scenes
    ```
    python tools/video_test.py --config configs/7scenes.yaml
    ```
  * Camrbrige
    ```
    python tools/video_test.py --config configs/cambridge.yaml
    ```

* Train

  We use ResNet-FPN pretrained [model](https://drive.google.com/file/d/1SJBjPx82FsDS84yaVstdjmPzYWAyliD8/view?usp=sharing).
  ```
    python tools/train_net.py
  ```
