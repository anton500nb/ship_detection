## Unet Convolutional Neural Network Model for Ship Detection from Satellite Images.
### Dataset taken from Kaggle.

The dataset contains 768x768 images and a CSV file with masks for each ship in the image.

There are multiple masks for each image and images without ships.

The data set is splited into training, validation and test parts.

In training and validation sets removed empty rows because this data not useful for training the Model.

In test set images without ships wasn't removed.

Convolutional neural network architecture: 41 layers, four blocks of Maxpooling and Upsampling with four Concatenations. Input shape is 256x256, optimazer 'Adam' with 0,0001 lerning rate, loss - 'binary_crossentropy', The Dice-score and f1-score as metric.

For today the best f1-score is 0.90, the Dice-score is 0.87.

### Main files:
1. airbus_ship_detection_custom_unet.ipynb 
2. model.py
3. inference.py
4. requirements.txt

### How to reproduce results:
1. Download dataset from Kaggle using:
  * !pip install kaggle
  * !mkdir -p ~/.kaggle
  * !echo '{"username":"input_your_username_here","key":"input_your_key here"}' > ~/.kaggle/kaggle.json
  * !chmod 600 ~/.kaggle/kaggle.json
  * !kaggle competitions download -c airbus-ship-detection --force
2. put all data from Kaggle to directiory with files;
3. to train the model run _model.py_;
4. to check the model on testing datset run _inference.py_;
5. to skip fit model you can download my _last_saved_model.h5_ file and put it into your dir:
  * https://drive.google.com/file/d/1NyG4S36C4PInhCGfJoV5GVV6eokOHyCP/view?usp=sharing .
