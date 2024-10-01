## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, PowerTransformer
data = pd.read_csv('/content/Encoding Data.csv')
data
```
![image](https://github.com/user-attachments/assets/48a62523-6368-492c-8d80-1b689d32110e)
```
# binary encoding
data['bin_1'] = data['bin_1'].map({'F': 0, 'T': 1})
data['bin_2'] = data['bin_2'].map({'N': 0, 'Y': 1})

# Ordinal Encoding
data['ord_2'] = data['ord_2'].astype(str)
ordinal_encoder = OrdinalEncoder(categories=[['Cold', 'Warm', 'Hot']])
data['ord_2'] = ordinal_encoder.fit_transform(data[['ord_2']])
data

```
![image](https://github.com/user-attachments/assets/10a04435-322d-4f57-9adf-d9d684487df0)
```
# One Hot Encoding for 'nom_0'
one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
nom_0_encoded = one_hot_encoder.fit_transform(data[['nom_0']])
nom_0_encoded_df = pd.DataFrame(nom_0_encoded, columns=one_hot_encoder.get_feature_names_out(['nom_0']))
data = pd.concat([data, nom_0_encoded_df], axis=1).drop('nom_0', axis=1)
data
```
![image](https://github.com/user-attachments/assets/871e4813-449c-491a-bde0-92c1f3e437c5)
```
# STEP 3: Apply Feature Transformation
# Using PowerTransformer for normalizing numeric columns
power_transformer = PowerTransformer(method='yeo-johnson')
data[['bin_1', 'bin_2', 'ord_2']] = power_transformer.fit_transform(data[['bin_1', 'bin_2', 'ord_2']])

data
```
![image](https://github.com/user-attachments/assets/85e8ee35-c4eb-4c46-81f8-cd1cb8875946)
```
# STEP 4: Save the processed data to a file
output_path = '/mnt/data/Processed_Encoding_Data.csv'
data.to_csv(output_path, index=False)
```
# RESULT:

Thus the program to read the given data and perform Feature Encoding and Transformation process and save the data to a file is successfully executed

       
