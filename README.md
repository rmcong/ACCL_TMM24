# Modeling Inner- and Cross-Task Contrastive Relations for Continual Image Classification

The implementation of the paper [Modeling Inner- and Cross-Task Contrastive Relations for Continual Image Classification](https://ieeexplore.ieee.org/document/10557156)

_Abstract_:
Existing continual image classification methods demonstrate that samples from all sequences of continual classification tasks contain common (task-invariant) features and class-specific (task-variant) features that can be decoupled for classification tasks. However, the existing feature decomposition strategies only focus on individual tasks while neglecting the essential cues that the relationship between different tasks can provide, thereby hindering the improvement of continual image classification results. To address this issue, we propose an Adversarial Contrastive Continual Learning (ACCL) method that decouples task-invariant and task-variant features by constructing all-round, multi-level contrasts on sample pairs within individual tasks or from different tasks. Specifically, three constraints on the distribution of task-invariant and task-variant features are included, i.e., task-invariant features across different tasks should remain consistent, task-variant features should exhibit differences, and task-invariant and task-variant features should differ from each other. At the same time, we also design an effective contrastive replay strategy to make full use of the replay samples to participate in the construction of sample pairs, further alleviating the forgetting problem, and modeling cross-task relationships. Through extensive experiments on continual image classification tasks on CIFAR100, MiniImageNet and TinyImageNet, we show the superiority of our proposed strategy, improving the accuracy and with better visualized outcomes.


## Installation
python==3.6.13
pytorch==1.9.0
CUDA=11.2

## Getting Start
### Project organization
```bash
└── ./data
    ├── miniimagenet
    │...
└──./networks
└──./checkpoints    
└──...
```
### Datasets
#### MiniImageNet
miniImageNet data should be [downloaded](https://github.com/yaoyao-liu/mini-imagenet-tools#about-mini-ImageNet) and pickled as a dictionary (data.pkl) with images and labels keys and placed in a sub-folder in './data' named as miniimagenet.

### Training
To train on MiniImageNet, run the following command:
```bash
python train.py
```
### Testing
We also privode the [trained model](https://portland-my.sharepoint.com/:u:/g/personal/yuxuanluo4-c_my_cityu_edu_hk/EdgIDLhrHO5EmkyL6boHJUwBpuPIjvxzETqj0cJW3XfwtQ?e=znqwWK) on MiniImageNet.
Download the model and place it in the './checkpoints' folder.
Then, to test the model, run the following command:
```bash
python test.py
```
## Citation
```
@ARTICLE{10557156,
  author={Luo, Yuxuan and Cong, Runmin and Liu, Xialei and Ip, Horace Ho Shing and Kwong, Sam},
  journal={IEEE Transactions on Multimedia}, 
  title={Modeling Inner- and Cross-Task Contrastive Relations for Continual Image Classification}, 
  year={2024},
  volume={},
  number={},
  pages={1-12},
  keywords={Task analysis;Feature extraction;Continuing education;Image classification;Training;Thermal stability;Stability analysis;continual learning;contrastive learning;feature decomposition;image classification},
  doi={10.1109/TMM.2024.3414277}}
```

## Acknowledgement
This code is based on the implementation of [ACL](https://github.com/facebookresearch/Adversarial-Continual-Learning?utm_source=catalyzex.com)
and [Co2L](https://github.com/facebookresearch/Adversarial-Continual-Learning?utm_source=catalyzex.com).
