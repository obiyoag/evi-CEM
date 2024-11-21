# Evidential Concept Embedding Models
Welcome👋! This repository provides the official implementation of our paper *Evidential Concept Embedding Models: Towards Reliable Concept Explanations for Skin Disease Diagnosis* (**evi-CEM**) [[arXiv](https://arxiv.org/abs/2406.19130)], which has been accepted by *MICCAI 2024*.

## 💡 TL;DR
![evi-cem](https://cdn.jsdelivr.net/gh/obiyoag/images@main/data/evi-cem.png)
> Concept Bottleneck Models (CBM) incorporate human-interpretable concepts into decision-making. However, their concept predictions may lack reliability when applied to clinical diagnosis. To address this, we propose an **evi**dential **C**oncept **E**mbedding **M**odel (**evi-CEM**), which employs evidential learning to model the concept uncertainty. Additionally, we offer to leverage the concept uncertainty to rectify concept misalignments that arise when training CBMs using vision-language models without complete concept supervision.

## 📦 Get started

### Environment Preparing
```
conda create -n evi-cem python=3.10
conda activate evi-cem
# please modify according to the CUDA version in your server
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Dataset Preparing
1. Run the following code to download `fitzpatrick17k.csv`
```
wget -P data/meta_data https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv
```
2. Run the following code to download the SkinCon Fitzpatrick17k annotation
```
wget -O data/meta_data/skincon.csv https://skincon-dataset.github.io/files/annotations_fitzpatrick17k.csv
```
3. Run `python data/raw_data_download.py` to download the fitzpatrick17k images. If any image links in the `fitzpatrick17k.csv` become invalid, the raw images can be downloaded [here](https://drive.google.com/file/d/1Eb7MGGr1Dj0z2xgEuMuCoblECuPDCrhD/view?usp=share_link)
4. Run `python data/generate_clip_concepts.py` to generate soft concept labels with [MONET](https://github.com/suinleelab/MONET)

### evi-CEM training

**Under complete concept supervision**:
```
python train.py --config configs/default.yaml
```
**Label-efficient training**:
```
python train.py --config configs/label_efficient.yaml
```
**Label-efficient training with concept rectification**:
```
python learn_cavs.py --config configs/learn_cavs.yaml
python train.py --config configs/train_rectified.yaml
```
## 🙋 Feedback and Contact
- ybgao22@m.fudan.edu.cn
- zxh@fudan.edu.cn

## 🛡️ License
This project is under the Apache-2.0 license. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgement
- We are grateful to [cem](https://github.com/mateoespinosa/cem) for publicly sharing their code, which we have drawn upon in our work.
- We appreciate the open-source dermatology dataset [Fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k) and the [SkinCon concept annotations](https://skincon-dataset.github.io) for it.
- We appreciate the effort of [MONET](https://github.com/suinleelab/MONET) to train and release the dermatology vision-language model.

## 📝 Citation
If our work or code is helpful in your research, please star this repo and cite our paper as follows.
```
@inproceedings{Gao2024eviCEM,
    author={Yibo Gao, Zheyao Gao, Xin Gao, Yuanye Liu, Bomin Wang, Xiahai Zhuang},
    title={Evidential Concept Embedding Models: Towards Reliable Concept Explanations for Skin Disease Diagnosis},
    booktitle={International Conference on Medical Image Computing and Computer Assisted Intervention},
    year={2024}
}
```
