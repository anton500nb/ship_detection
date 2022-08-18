## Unet custom Convolutional Neural Network Model for Ship Detection from Satellite Images.
### Dataset taken from Kaggle.

The dataset contains 768x768 images and a CSV file with masks for each ship in the image.

There are multiple masks for each image and images without ships.

The dataset is splited into training, validation and test parts.

In training set removed empty rows because this data not useful for training the Model.

In test set images without ships wasn't removed.

Input shape images is 256x256, optimazer 'Adam' with 0,0001 lerning rate,
loss - 'binary_crossentropy', The Dice-score as a metric.

##### For today the best f1-score is 0.75 and the best Dice-score is 0.5.

#### Main files:
1. airbus_ship_detection_custom_unet.ipynb 
2. model.py
3. inference.py
4. requirements.txt

### How to reproduce results:

#### Download dataset from Kaggle using:

1. !pip install kaggle

2. !mkdir -p ~/.kaggle

3. !echo '{"username":"input_your_username_here","key":"input_your_key here"}' > ~/.kaggle/kaggle.json

4. !chmod 600 ~/.kaggle/kaggle.json

5. !kaggle competitions download -c airbus-ship-detection --force

#### Put all data from Kaggle to directiory with files.

1. To train the model run model.py.
2. To check the model on testing datset run inference.py.

#### For skipping the Model training, you can download my h5 file here:

https://drive.google.com/file/d/1NyG4S36C4PInhCGfJoV5GVV6eokOHyCP/view?usp=sharing

and put it in your dir.
