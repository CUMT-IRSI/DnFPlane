# DnFPlane:DnFPlane For Efficient and High-Quality 4D Reconstruction of Deformable Tissues（MICCAI2024）

### [Paper]()[Paper-2081.pdf](https://github.com/user-attachments/files/16115503/Paper-2081.pdf)


> [DnFPlane For Efficient and High-Quality 4D Reconstruction of Deformable Tissues]() \
> Ran Bu, Chenwei Xu, Jiwei Shan, Hao Li, Guangming Wang, Yanzi Miao, Hesheng Wang \
>  R. Bu and C. Xu—Equal contribution. \
> MICCAI2024

![backbone](https://github.com/CUMT-IRSI/DnFPlane/assets/174862475/0384128b-d48f-4e37-bb34-30259df6bda7)

## Schedule
- [x] Initial Code Release.
- [ ] Further check of the reproducibility.
- [ ] The code release for the extended version (with StereoMIS Dataset).

## Introduction
Reconstruction of deformable tissues in robotic surgery from endoscopic stereo videos holds great significance for a variety of clinical applications. Existing methods primarily focus on enhancing inference speed, overlooking depth distortion issues in reconstruction results, particularly in regions occluded by surgical instruments. This may lead to misdiagnosis and surgical misguidance. In this paper, 
we propose an efficient algorithm designed to address the reconstruction challenges arising from depth distortion in complex scenarios. Unlike previous methods that treat each feature plane equally in the dynamic and static field, our framework guides the static field with the dynamic field, generating a dynamic-mask to filter features at the time level. This allows the network to focus on more active dynamic features, reducing depth distortion. In addition, we design a module to address dynamic blurring. Using the dynamic-mask as a guidance, we iteratively refine color values through Gated Recurrent Units (GRU), improving the clarity of tissues detail in the reconstructed results. Experiments on a public endoscope dataset demonstrate that our method outperforms existing state-of-the-art methods without compromising training time. Furthermore, our approach shows outstanding reconstruction performance in occluded regions, making it a more reliable solution in medical scenarios.
## DnFPlane VS LerPlane in terms of rendering quality



https://github.com/CUMT-IRSI/DnFPlane/assets/174862475/b81ddb5a-ec73-42b6-9d73-a84d37ea024b



## Installation

### Set up the Python environment
<details> <summary>Tested with an Ubuntu workstation NVIDIA RTX 3090GPU.</summary>

```
conda create -n dnfplane python=3.9
conda activate dnfplane
pip install -r requirements.txt
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch 
```
If you want to train/test our model on some latest GPUs like RTX4090(tested 2023.12), you can follow this project(https://github.com/Loping151/ForPlane).
</details>

### Set up datasets
<details> <summary>Download the datasets</summary> 

Please download the dataset from [EndoNeRF](https://github.com/med-air/EndoNeRF) 

To use the example config, organize your data like:
```
data
| - endonerf_full_datasets
|   | - cutting_tissues_twice
|   | - pushing_soft_tissues
| - StereoMIS
|   | - stereo_seq_1
|   | - stereo_seq_1
```

</details>

### training
<details> <summary>Using configs for training</summary> 

dnfplane uses configs to control the training process. The example configs are stored in the `dnfplanes/config` folder.
To train a model, run the following command:
```
export CUDA_VISIBLE_DEVICES=0
PYTHONPATH=. python dnfplanes/main.py --config-path /data/DnFPlane/dnfplanes/config/example_cutting-9k.py
```
</details>

### Evaluation
We use the same evaluation protocol as [EndoNeRF](https://github.com/med-air/EndoNeRF). So please follow the instructions in EndoNeRF.

## Acknowledgements
We would like to acknowledge the following inspiring work:
- [EDSSR](https://arxiv.org/pdf/2107.00229) (Long et al.)
- [EndoNeRF](https://github.com/med-air/EndoNeRF) (Wang et al.)
- [K-Planes](https://sarafridov.github.io/K-Planes/) (Fridovich-Keil et al.)
- [LerPlane](https://github.com/Loping151/ForPlane) (Yang et al.)



