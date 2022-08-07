import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import keras
import segmentation_models as sm
sm.set_framework('tf.keras')
from segmentation_models import Unet
from segmentation_models.metrics import FScore
from keras.optimizers import adam_v2



# Load data

train_df = pd.read_csv('train_df.csv')
val_df = pd.read_csv('val_df.csv')
print('CSV file loaded')



# Replace NaN values with empty strings in test dataframe
# because images without ships will be helpful for validation and tests.

val_df = val_df.fillna('') 



# Remove the NaN values in train because for the training model it does not need it.

train_df = train_df.dropna(axis = 'index')



# Function fo decoding lables to masks

def rle_decode(mask_rle, shape=(768, 768)):            
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T 



# Generators for Model training (to reduce memory usage)

def img_generator(gen_df, batch_size):                            
    while True:
        x_batch = []
        y_batch = []
        
        for i in range(batch_size):
            img_name, mask_rle = gen_df.sample(1).values[0]      # get row from DF
            img = cv2.imread('train_v2/'+ img_name)              # read img
            

            img_masks = train_df.loc[train_df['ImageId'] == img_name, 'EncodedPixels'].tolist()

            all_masks = np.zeros((768, 768))                     # find ship masks for more the one ship                   
            for mask in img_masks:                               # create a single mask for all ships
                all_masks += rle_decode(mask)
            

            img = cv2.resize(img, (256, 256))                    # resize img to 256,256
            mask = cv2.resize(all_masks, (256, 256))             # resize mask to 256,256
            
            
            x_batch += [img]                                     # put into batch
            y_batch += [mask]

        x_batch = np.array(x_batch) / 255.                       # reduce color dimension img
        y_batch = np.array(y_batch)

        yield x_batch, np.expand_dims(y_batch, -1)               # return batch



# # Generators for Model testing and evaluation (to reduce memory usage)       
        
def img_generator_test(gen_df, batch_size):                      
    while True:                                                  
        x_batch = []                                             
        y_batch = []
        
        for i in range(batch_size):
            img_name, mask_rle = gen_df.sample(1).values[0]      # get row from DF
            img = cv2.imread('train_v2/'+ img_name)              # read img
            
            mask = rle_decode(mask_rle)                          # decode lable to mask
            
            img = cv2.resize(img, (256, 256))                    # resize it to 256,256
            mask = cv2.resize(mask, (256, 256))
            
            
            x_batch += [img]                                     # put into batch
            y_batch += [mask]

        x_batch = np.array(x_batch) / 255.                       # reduce color dimension img
        y_batch = np.array(y_batch)

        yield x_batch, np.expand_dims(y_batch, -1)               # return batch



# Make the Model

model = Unet(backbone_name = 'resnet34',                          # net name
             input_shape=(256, 256, 3),
             classes = 1,                                         # to find only one class - mask
             encoder_weights = 'imagenet',
             encoder_freeze = True,
             activation = 'sigmoid' )                             # for probability of belonging to a ship 0-1


best_w = keras.callbacks.ModelCheckpoint('r34_best.h5',           # save best weights during training
                                monitor = 'val_loss',
                                verbose = 0,
                                save_best_only = True,
                                save_weights_only = True,
                                mode = 'auto',
                                period = 1)

last_w = keras.callbacks.ModelCheckpoint('r34_last.h5',            # save last weights during training
                                monitor = 'val_loss',
                                verbose = 0,
                                save_best_only = False,
                                save_weights_only = True,
                                mode='auto',
                                period=1)


callbacks = [best_w, last_w]



adam = tf.keras.optimizers.Adam(learning_rate = 0.0001,
                                beta_1 = 0.9,
                                beta_2 = 0.999,
                                epsilon = 1e-08,
                                decay = 0.0)


metric = FScore()                                                   # Dison metric

model.compile(adam, 'binary_crossentropy', [metric])

batch_size = 16



print ('Training is started')

history = model.fit(img_generator(train_df, batch_size),
              steps_per_epoch = 100,
              epochs = 300,
              verbose = 1,
              callbacks = callbacks,
              validation_data = img_generator_test(val_df, batch_size),
              validation_steps = 10,
              class_weight = None,
              max_queue_size = 10,
              workers = 1,
              use_multiprocessing = False,
              shuffle = True,
              initial_epoch = 0)



# Save the Model

model.save('last_saved_model.h5')                                    
print('Training completed successfully and the Model has been saved to a file "last_saved_model.h5"')


