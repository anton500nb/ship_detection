import pandas as pd
import numpy as np
from keras.models import load_model
import cv2
import tensorflow as tf
import keras
import segmentation_models as sm
sm.set_framework('tf.keras')
from segmentation_models import Unet
from segmentation_models.metrics import FScore



# Load the Model

model = load_model('last_saved_model.h5', custom_objects={'f1-score': sm.metrics.FScore()})

print('The Model loaded')


# Load data

masks = pd.read_csv('train_ship_segmentations_v2.csv')
train_df = masks[:230000]
test_df = masks[230000:]
test_df.reset_index(drop = True, inplace = True)

print('CSV file loaded')



# Decoding mask from dataset function 

def rle_decode(mask_rle, shape=(768, 768)):            
    s = str(mask_rle).split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T 



# Generator 'img + mask' for testing 

def img_generator(gen_df, batch_size):                            
    while True:
        x_batch = []
        y_batch = []
        
        for i in range(batch_size):
            img_name, mask_rle = gen_df.sample(1).values[0]      # get row from DF
            img = cv2.imread('train_v2/'+ img_name)              # read img
            

            img_masks = masks.loc[masks['ImageId'] == img_name, 'EncodedPixels'].tolist()

            all_masks = np.zeros((768, 768))                     # find ship masks for more the one ship                   
            for mask in img_masks:                               # create a single mask for all ships
                if type(mask) == str:
                    all_masks += rle_decode(mask) 
            

            img = cv2.resize(img, (256, 256))                    # resize img to 256,256
            mask = cv2.resize(all_masks, (256, 256))             # resize mask to 256,256
            
            
            x_batch += [img]                                     # put into batch
            y_batch += [mask]

        x_batch = np.array(x_batch) / 255.                       # reduce color dimension img
        y_batch = np.array(y_batch)

        yield x_batch, np.expand_dims(y_batch, -1)               # return batch

        

# Make prediction

for x, y in img_generator(test_df, 100):                                                 
    break

y = np.float32(y)        # to reduce memory using
pred = model.predict(x)



# Evaluation the Model

print('Coef MeanIoU: ', tf.keras.metrics.MeanIoU(num_classes=2)(y, pred))
print('The F-score (Dice coefficient): ', FScore()(y, pred))
