<div align="center">

<h3>XHand: Real-time Expressive Hand Avatar</h3>

[Qijun Gan](https://github.com/agnJason), Zijie Zhou, [Jianke Zhu](https://scholar.google.cz/citations?user=SC-WmzwAAAAJ)<sup>:email:</sup>
 
Zhejiang University

(<sup>:email:</sup>) corresponding author.

[ArXiv Preprint](https://arxiv.org/abs/2407.21002) &nbsp;&nbsp;&nbsp;&nbsp; [Project Page](https://agnjason.github.io/XHand-page/) 

</div>

### News
* `Jul. 31st, 2024`: 🔥 We have released our code. Try it now! Please give us a star! ⭐️⭐️⭐️ 😄
* Our source code is coming soon. Please stay tuned! ☕️

## Abstract

Hand avatars play a pivotal role in a wide array of digital interfaces, enhancing user immersion and facilitating natural interaction within virtual environments. While previous studies have focused on photo-realistic hand rendering, little attention has been paid to reconstruct the hand geometry with fine details, which is essential to rendering quality. In the realms of extended reality and gaming, on-the-fly rendering becomes imperative. To this end, we introduce an expressive hand avatar, named XHand, that is designed to comprehensively generate hand shape, appearance, and deformations in real-time. To obtain fine-grained hand meshes, we make use of three feature embedding modules to predict hand deformation displacements, albedo, and linear blending skinning weights, respectively. To achieve photo-realistic hand rendering on fine-grained meshes, our method employs a mesh-based neural renderer by leveraging mesh topological consistency and latent codes from embedding modules. During training, a part-aware Laplace smoothing strategy is proposed by incorporating the distinct levels of regularization to effectively maintain the necessary details and eliminate the undesired artifacts. The experimental evaluations on InterHand2.6M and DeepHandMesh datasets demonstrate the efficacy of XHand, which is able to obtain high-fidelity geometry and texture for hand animations across diverse poses in real-time.

## Introduction

![framework](assets/teaser.png "framework")
<div align="center-align">We present XHand, a rigged hand avatar that captures the  geometry, appearance and poses of the hand. XHand is created from multi-view videos and utilizes MANO pose parameters (the first image in each group of (a)) to generate high-detail meshes (the second) and renderings (the third). XHand generates photo-realistic hand images in real-time for a given pose sequence (b). (c) is an example of animated personalized hand avatars according to poses in the wild images.</div>

**Notes**: 

- All the experiments are performed on 1 NVIDIA GeForce RTX 3090Ti GPU.


## Getting Started

### Install 

**a. Create a conda virtual environment and install required packages.**
```shell
git clone git@github.com:agnJason/XHand.git
conda create -n xhand python=3.10 -y
conda activate xhand

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

**b. Prepare MANO models.**

Besides, you also need to download the MANO model. Please visit the [MANO website](https://mano.is.tue.mpg.de/) and register to get access to the downloads section. You need to put MANO_RIGHT.pkl and MANO_LEFT.pkl under the ./mano folder.

### Training
Edit your Interhand2.6M PATH in [conf/ih_sfsseq.conf](conf/ih_sfsseq.conf)->data_path, which should contain ./images and ./annotations.
```bash
python sfs_lbs_train.py --conf conf/ih_sfsseq.conf
```
The output should be in `./interhand_out`.

### Inference
[Pretrained model](https://drive.google.com/file/d/1xDtMKb08aNDarDPSBOQzpTfXrF8aR5uA/view?usp=sharing)
```bash
python sfs_lbs_test.py --exp_path interhand_out/Capture0_ROM03_RT_No_Occlusion/xhand
```
Try `--cam_id (cam400004)`, `--test_data_name (0002_good_luck)` and `--test_capture_name (Capture1)` tu inference with different views or poses. Save visualized results with `--save_vis` and save detailed meshes with `--save_mesh`.

## Citation
If you find our project is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.
```bibtex
@misc{gan2024xhandrealtimeexpressivehand,
      title={XHand: Real-time Expressive Hand Avatar}, 
      author={Qijun Gan and Zijie Zhou and Jianke Zhu},
      year={2024},
      eprint={2407.21002},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.21002}, 
}
```
