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
```
We follow the process described in [1,2]. The code of incorporating image features are mainly borrowed from [1](https://github.com/j-min/VL-T5).
```

# Traing
For multi-gpu training (8 gpus), run it like this: 
```
bash multimodal_dist_mmt5_32_ft.sh 4 11 mm-sum 1.0 8 256  
```
For single-gpu training, run it like this: 
```
bash single_lang_multimodal_train32.sh english # e.g., for training on english dataset.
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
