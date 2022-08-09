Unet Convolutional Neural Network Model for Ship Detection from Satellite Images.
Dataset taken from Kaggle.
The dataset contains 768x768 images and a CSV file with masks for each ship in the image.
There are multiple masks for each image and images without ships.
The data set is splited into training, validation and test parts.
In training and validation sets removed empty rows because this data not useful for training the Model.
In test set images without ships wasn't removed.
Images before load to the Model croped and augmented.
U-net with Resnet34 encoder was chosen to train the model
from Segmentation Models Python API.
Input shape is 256x256, optimazer 'Adam' with 0,0001 lerning rate,
loss - 'binary_crossentropy', The F-score (Dice coefficient) as metric.
900 epochs with 100 steps/per epoch (bath size 16).

For today the best F-score is 0.64.
The public Kaggle Score is 0.76826.
For the best result need to apply augmentation.

Main files:
airbus_ship_detection_resnet34.ipynb 
model.py
inference.py
last_saved_model.h5

How to reproduce results:
Download dataset from Kaggle using:
!pip install kaggle
!mkdir -p ~/.kaggle
!echo '{"username":"input_your_username_here","key":"input_your_key here"}' > ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c airbus-ship-detection --force
Put all data from Kaggle to directiory with files.

To train the model run model.py.
To check the model on testing datset run inference.py.