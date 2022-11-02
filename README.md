# CLIP + SVM Image Classifier

- CLIP('RN50x16') 모델의 vision 부분으로 768 dim의 feature 벡터들을 추출한 후
- feature 벡터와 label을 활용하여 SVM(kernel='rbf') 모델을 학습시켜
- multi-class Image Classification을 진행하였습니다.


## DATASET
[Kaggle Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- class: buildings, forest, glacier, mountain, sea, street
- test accuracy: 95.06
- macro f1: 95.15
- 데이터셋은 Kaggle에서 다운로드 후 dataset 폴더 하위에 'seg_train', 'seg_test'를 unzip 하시면 됩니다.