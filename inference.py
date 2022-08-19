import pandas as pd
import numpy as np
from keras.models import load_model
import cv2
import tensorflow as tf
import keras
from keras import backend as K


# Metrics

def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())
    return recall_keras


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_keras = true_positives / (predicted_positives + K.epsilon())
    return precision_keras


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))



# Load the Model

model = load_model('last_saved_model.h5', custom_objects={'dice_coeff': dice_coeff, 'f1': f1})

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

for x, y in img_generator(test_df, 1000):                                                 
    break

y = np.float32(y)        # to reduce memory using
pred = model.predict(x)



# Evaluation the Model

print('The f1-score: ', f1(y, pred))
print('The Dice coefficient: ', dice_coeff(y, pred))
