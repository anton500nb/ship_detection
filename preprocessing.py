import pandas as pd
from sklearn.model_selection import train_test_split



# Load CSV file

masks = pd.read_csv('train_ship_segmentations_v2.csv')

# The dataframe has 230k+ rows and 150k NaN values (img without ships).



# Split data for train:validation:test 70:29:0.5.

train_df = masks[:200000]
test_df = masks[200000:]
test_df.reset_index(drop=True, inplace=True)
test_df, val_df = train_test_split (test_df, test_size=0.95, random_state=42)

# Remove the NaN values in train because for the training model it does not need it.

train_df = train_df.dropna(axis='index')

# Save all dataframes

pd.DataFrame(train_df).to_csv('train_df.csv', header = True , index = False) 
pd.DataFrame(val_df).to_csv('val_df.csv', header = True , index = False) 
pd.DataFrame(test_df).to_csv('test_df.csv', header = True , index = False) 

print('The dataset is preprocessed and split.')