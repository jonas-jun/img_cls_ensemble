# CLIP + SVM Image Classifier

- CLIP('RN50x16') 모델의 vision 부분으로 768 dim의 feature 벡터들을 추출한 후
- feature 벡터와 label을 활용하여 SVM(kernel='rbf') 모델을 학습시켜
- multi-class Image Classification을 진행하였습니다.
- feature의 dimension이 커질수록 sparse한 경향이 있어 SVM 분류에서 성능이 떨어질 수 있어, 768 dim의 feature extractor를 선택하였습니다.

## DATASET
[Kaggle Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- class: buildings, forest, glacier, mountain, sea, street
- **test accuracy: 95.06** (공개된 notebook code에서는 95% 이상의 acc를 보이는 경우가 흔치 않았습니다)
- macro f1: 95.15
- 데이터셋은 Kaggle에서 다운로드 후 dataset 폴더 하위에 'seg_train', 'seg_test'를 unzip 하시면 됩니다.
