import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Dense, Concatenate
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, UpSampling2D, Conv2D, MaxPooling2D, Activation, Dropout
from keras.optimizers import adam_v2
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



# Load data

masks = pd.read_csv('train_ship_segmentations_v2.csv')

print('CSV file loaded')



# Split data to train, validation and test sets (70:28:2)

train_df = masks[:230000]
test_df = masks[230000:]
test_df.reset_index(drop = True, inplace = True)
train_df = train_df.dropna(axis = 'index')       # images without ships don't need for train
train_df, val_df = train_test_split (train_df, test_size = 0.3, random_state = 42)



# Function fo decoding lables to masks

def rle_decode(mask_rle, shape = (768, 768)):            
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype = np.uint8)

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
            

            img_masks = masks.loc[masks['ImageId'] == img_name, 'EncodedPixels'].tolist()

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



# Make the Model

inp = Input(shape=(256, 256, 3))                               # input layer with shape 256x256 and 3 chanels



conv_1_1 = Conv2D(32, (3, 3), padding = 'same')(inp)           # increase filters number to 32, kernel size 3x3
conv_1_1 = Activation('relu')(conv_1_1)

conv_1_2 = Conv2D(32, (3, 3), padding = 'same')(conv_1_1)
conv_1_2 = Activation('relu')(conv_1_2)

pool_1 = MaxPooling2D(2)(conv_1_2)                             # reduce image size 



conv_2_1 = Conv2D(64, (3, 3), padding = 'same')(pool_1)        # increase filters number to 64
conv_2_1 = Activation('relu')(conv_2_1)

conv_2_2 = Conv2D(64, (3, 3), padding = 'same')(conv_2_1)
conv_2_2 = Activation('relu')(conv_2_2)

pool_2 = MaxPooling2D(2)(conv_2_2)                             # reduce image size 



conv_3_1 = Conv2D(128, (3, 3), padding = 'same')(pool_2)       # increase filters number to 128
conv_3_1 = Activation('relu')(conv_3_1)

conv_3_2 = Conv2D(128, (3, 3), padding = 'same')(conv_3_1)
conv_3_2 = Activation('relu')(conv_3_2)

pool_3 = MaxPooling2D(2)(conv_3_2)                             # reduce image size



conv_4_1 = Conv2D(256, (3, 3), padding = 'same')(pool_3)       # increase filters number to 256
conv_4_1 = Activation('relu')(conv_4_1)

conv_4_2 = Conv2D(256, (3, 3), padding = 'same')(conv_4_1)
conv_4_2 = Activation('relu')(conv_4_2)

pool_4 = MaxPooling2D(2)(conv_4_2)                             # reduce image size



up_1 = UpSampling2D(2, interpolation = 'bilinear')(pool_4)     # increase image size

conc_1 = Concatenate()([conv_4_2, up_1])                       # concatenate withe the same size layer before upsampling to get low level info

conv_up_1_1 = Conv2D(256, (3, 3), padding = 'same')(conc_1)
conv_up_1_1 = Activation('relu')(conv_up_1_1)

conv_up_1_2 = Conv2D(256, (3, 3), padding = 'same')(conv_up_1_1)
conv_up_1_2 = Activation('relu')(conv_up_1_2)



up_2 = UpSampling2D(2, interpolation = 'bilinear')(conv_up_1_2) # increase image size

conc_2 = Concatenate()([conv_3_2, up_2])                        # concatenate withe the same size layer before upsampling to get low level info

conv_up_2_1 = Conv2D(128, (3, 3), padding = 'same')(conc_2)     # reduce filter number to 128
conv_up_2_1 = Activation('relu')(conv_up_2_1)

conv_up_2_2 = Conv2D(128, (3, 3), padding = 'same')(conv_up_2_1)
conv_up_2_2 = Activation('relu')(conv_up_2_2)



up_3 = UpSampling2D(2, interpolation = 'bilinear')(conv_up_2_2) # increase image size

conc_3 = Concatenate()([conv_2_2, up_3])                        # concatenate withe the same size layer before upsampling to get low level info

conv_up_3_1 = Conv2D(64, (3, 3), padding = 'same')(conc_3)      # reduce filter number to 64
conv_up_3_1 = Activation('relu')(conv_up_3_1)

conv_up_3_2 = Conv2D(64, (3, 3), padding='same')(conv_up_3_1)
conv_up_3_2 = Activation('relu')(conv_up_3_2)



up_4 = UpSampling2D(2, interpolation = 'bilinear')(conv_up_3_2) # increase image size

conc_4 = Concatenate()([conv_1_2, up_4])                        # concatenate withe the same size layer before upsampling to get low level info

conv_up_4_1 = Conv2D(32, (3, 3), padding = 'same')(conc_4)      # reduce filter number to 32
conv_up_4_1 = Activation('relu')(conv_up_4_1)

conv_up_4_2 = Conv2D(1, (3, 3), padding = 'same')(conv_up_4_1)
conv_up_4_3 = Dropout(0.5)(conv_up_4_2)                         # avoid overfitting



result = Activation('sigmoid')(conv_up_4_3)                     # otput layer with sigmoid activation to get probability is a pixel ship



model = Model(inputs = inp, outputs = result)


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


model.compile(adam, 'binary_crossentropy', [f1, dice_coeff])

batch_size = 16



print ('Training is started')

model.fit(img_generator(train_df, batch_size),
              steps_per_epoch = 100,
              epochs = 300,
              verbose = 1,
              callbacks = callbacks,
              validation_data = img_generator(val_df, batch_size),
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
