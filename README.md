# [JBHI 2026] Dual-Level Imbalance Mitigation for Single-FoV Colorectal Histopathology Image Classification


## Authors

Lingling Yuan, Yang Chen, Md Rahaman, Hongzan Sun, Xiaoyan Li, Haoyuan Chen, Yutong Gu, Mengqing Su, Marcin Grzegorzek, Chen Li



## Abstract

Single-field-of-view (FoV) histopathological image classification is vital for colorectal cancer (CRC) diagnosis in mid- to low-tier hospitals lacking whole-slide imaging (WSI) scanners and storage, yet suffers from severe class imbalance and degraded performance. To address this, we propose a dual-level imbalance mitigation (DIM) framework integrating data-level and algorithm-level approaches. Specifically:  
1. **Data-level approach:** A global context generative adversarial network (GCGAN) generates realistic minority-class images for augmentation to balance the dataset.  
2. **Algorithm-level approach:** A frequency-aware adaptive focal loss (FAFL) applies a frequency-aware offset and adaptive modulation to better separate overlapping classes.  
3. **Classification model:** A lightweight receptive field-based convolutional neural network (LRF-CNN) is trained under DIM to leverage both augmentation and loss modulation for improved classification.

Extensive experiments on single-FoV CRC datasets demonstrate that DIM-equipped LRF-CNN outperforms five state-of-the-art models (SOTA) across multiple metrics. Furthermore, each DIM component enhances performance when applied individually to those models, and additional validation on diverse histopathological datasets confirms the generalizability and effectiveness of the proposed DIM framework. 




  
![Overview](Fig-overview.png)



## Environment Setup

### Classification Task Environment


This project is developed with **Python 3.8.18** and **PyTorch 1.13.1+cu117**. Follow the steps below to create a new conda environment named **DIM** and install the necessary dependencies:

```bash
conda create --name DIM python=3.8.18 -y
conda activate DIM
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
 ``` 



## Dataset and Pre-trained Weights

The dataset partitioning strategy and pre-trained weights used in our experiments can be obtained upon request. For more details regarding the dataset splits and accessing the pre-trained models, please contact the first author:

**Email:** [yuanlingling0314@163.com](mailto:yuanlingling0314@163.com)



