# fast-food-classification

Training a nn to classify food items using PyTorch.
Trying to follow Andrej Karpathy's [recipe](http://karpathy.github.io/2019/04/25/recipe/) for training nn.

Ended up fine-tuning an EfficientNetB0, pre-trained on ImageNet.

## EDA

### Training Data

![](./readme_media/training_data.png)

### Class Distribution

![](./readme_media/count_plot.png)

### Image Size Distribution

![](./readme_media/size_distribution.png)

## Baseline

### Input Independent Test

![](./readme_media/input_independent_test.png)

### Single Batch Overfit Test

![](./readme_media/overfit_single_batch.png)

## Model tuning

### Prediction

![](./readme_media/predictions.png)

### Results

|       | Accuracy | Loss   |
| ----- | -------- | ------ |
| Train | 97.55%   | 0.0764 |
| Test  | 88.92%   | 0.2931 |
