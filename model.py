import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import albumentations as A
import tensorflow as tf
import keras
import segmentation_models as sm
sm.set_framework('tf.keras')
from segmentation_models import Unet
from segmentation_models.metrics import FScore
from keras.optimizers import adam_v2



# Load data

masks = pd.read_csv('train_ship_segmentations_v2.csv')

# The dataframe has 230k+ rows and 150k NaN values (img without ships)
# Split data for train:validation:test 70:28:2

train_df = masks[:220000]
test_df = masks[220000:]
test_df.reset_index(drop=True, inplace=True)
train_df = train_df.dropna(axis='index')    # images without ships don't need for train
train_df, val_df = train_test_split (train_df, test_size=0.3, random_state=42)


# Save dataframe for test

pd.DataFrame(test_df).to_csv('test_df.csv', header = True , index = False) 
print('CSV file loaded')



# Function for decoding lables to masks

def rle_decode(mask_rle, shape=(768, 768)):            
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T 



#Function for augmentation images

transform = A.Compose([
    A.ShiftScaleRotate(shift_limit = 0.125, scale_limit=0.2, rotate_limit = 10, p = 0.7, border_mode = cv2.BORDER_CONSTANT),
    A.RandomCrop(256, 256),
    A.RandomRotate90(p = .2),
    A.ElasticTransform(1., p = .2),
    A.HorizontalFlip(p = .2),
    A.OneOf([A.RandomCrop(256, 256),
            A.GaussNoise( ),
            ], p = 0.3),
    A.OneOf([  A.RandomCrop(256, 256),
            A.MotionBlur(p = .4),
            A.Blur(blur_limit = 3, p = 0.3),
        ], p = 0.5),
        A.OneOf([ A.RandomCrop(256, 256),
            A.OpticalDistortion(p = 0.3),
            A.GridDistortion(p = 0.1),
            A.PiecewiseAffine(p = 0.3),
        ], p = 0.5),
        A.OneOf([ A.RandomCrop(256, 256),
            A.CLAHE(clip_limit = 3),
            A.Sharpen(),
            A.Emboss(),
            
        ], p = 0.4),
], p = 1)



# Generators images for Model (to reduce memory usage)

def img_generator(gen_df, batch_size):                      
    while True:
        x_batch = []
        y_batch = []
        
        for i in range(batch_size):
            img_name, mask_rle = gen_df.sample(1).values[0]                                 # get row from DF
            img = cv2.imread('train_v2/'+ img_name)                                         # read img
            
            img_masks = masks.loc[masks['ImageId'] == img_name, 'EncodedPixels'].tolist()   # find all ship masks for image with more the one ship                                          
            all_masks = np.zeros((768, 768))                                                # create a single mask array for all ships
            for mask in img_masks:
                all_masks += rle_decode(mask)
            
            transformed = transform(image = img, mask = all_masks)                          # augm. data crop, rotate ect.
            image_transformed = transformed['image']
            mask_transformed = transformed['mask']
            
            x_batch += [image_transformed]                                                  # put into batch
            y_batch += [mask_transformed]

        x_batch = np.array(x_batch) / 255.                                                  # reduce color dimension img
        y_batch = np.array(y_batch)

        yield x_batch, np.expand_dims(y_batch, -1)                                          # return batch


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


