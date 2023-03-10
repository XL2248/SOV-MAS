Our code is based on the [xl-sum](https://github.com/csebuetnlp/xl-sum) of huggingface transformers.

# The Dependency:
```
python==3.7.9
pytorch==1.7.1 
torchvision==0.8.2 
torchaudio==0.7.2 
cudatoolkit=10.2
```

# Visual Features Extraction and usage
The visual features extraction code are mainly from [image_feature_extraction](https://github.com/j-min/VL-T5/tree/main/feature_extraction) [1,2]. 

The code of incorporating image features are mainly borrowed from [Vg-gplms](https://github.com/hltchkust/vg-gplms).

# Data

All the triplet data <image urls, article, and summary> can be downloaded [here](https://drive.google.com/file/d/1h-vWFQaZyOu_jbr6thwUWbzW93fOke0i/view?usp=share_link). Note that the training data of zero-shot languages are not used under the zero-shot setting.

# Traing
For multi-gpu multilingual training (8 gpus), run it like this: 
```
bash multimodal_dist_mmt5_32_ft.sh 4 11 high-resource 1.0 8 256   # high-resource for reproducing Table 1.
```
For single-gpu single-language training, run it like this: 
```
bash single_lang_multimodal_train32.sh high-resource english # e.g., for training on english dataset.
```

# Testing
For testing, run it: 
```
bash evaluate.sh.
```

# Reference
```
[1] Jaemin Cho, Jie Lei, Hao Tan, and Mohit Bansal. Unifying vision-and-language tasks via text generation. In ICML, 2021: 1931–1942.
[2] Anderson P, He X, Buehler C, et al. Bottom-up and top-down attention for image captioning and visual question answering[C]. In CVPR. 2018: 6077-6086.
```

# Citation
```
@misc{https://doi.org/10.48550/arxiv.2212.07672,
  doi = {10.48550/ARXIV.2212.07672},
  url = {https://arxiv.org/abs/2212.07672},
  author = {Liang, Yunlong and Meng, Fandong and Xu, Jinan and Wang, Jiaan and Chen, Yufeng and Zhou, Jie},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Summary-Oriented Vision Modeling for Multimodal Abstractive Summarization},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
