# vehicale_no_plate_recognition using Pretrained VGG Model

The model includes a pretrained VGG19 as feature_extractor => dense(relu) => bi-lstms => dense(softmax) and [CTC loss](https://towardsdatascience.com/intuitively-understanding-connectionist-temporal-classification-3797e43a86c) layer.


# Data and Some outputs of the model
The dataset has cropped RGB image liscence plate and the vehicle_no as output. the dataset.csv file has info related to data. Some of the outputs of the model is given below:
![image](https://github.com/vikey725/vehicale_no_plate_recognition/blob/main/saved_img/pred_2.png)

# Dependencies
- tensorflow 2.0
- numpy
- pandas
- matplotlib
- sklearn
- albumentations
- Levenshtein

# Training

  ```python -m code.engine train```

# Testing

  ```python -m code.engine```

  
# Accuracy

The word accuracy is 75.38 %
The char accuracy is 94.83 %

# Reference

[https://keras.io/examples/vision/captcha_ocr/](https://keras.io/examples/vision/captcha_ocr/)
