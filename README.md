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
      

```python 
import pandas as pd
df=pd.read_csv('/content/Encoding Data.csv')
df
```

![alt text](<Screenshot 2024-09-24 113250.png>)


```python 
df.shape

```
![alt text](<Screenshot 2024-09-24 113340.png>)


```python 
df.info()

```

![alt text](<Screenshot 2024-09-24 113350.png>)


```python 
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[['ord_2']])
```

![alt text](<Screenshot 2024-09-24 113402.png>)


```python 
df['bo_2']=e1.fit_transform(df[['ord_2']])
df

```

![alt text](<Screenshot 2024-09-24 113416-1.png>)


```python 
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(df['ord_2'])
dfc
```

![alt text](<Screenshot 2024-09-24 113416.png>)


```python 
dfc=df.copy()

```


```python 
dfc['con_2']=le.fit_transform(df['ord_2'])
dfc
```


![alt text](<Screenshot 2024-09-24 113512.png>)

```python 
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```

![alt text](<Screenshot 2024-09-24 113809.png>)

```python 
pd.get_dummies(df2,columns=['nom_0'])

```

![alt text](<Screenshot 2024-09-24 113825.png>)


```python 

pip install --upgrade category_encoders
```



```python 

from category_encoders import BinaryEncoder
```



```python 
df=pd.read_csv('/content/data.csv')
df

```
![alt text](<Screenshot 2024-09-24 114309.png>)


```python 
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
sfb1=df.copy()
dfb
```
![alt text](<Screenshot 2024-09-24 114338.png>)



```python 
from category_encoders import TargetEncoder

```



```python 
te=TargetEncoder()
cc=df.copy()
new = te.fit_transform(X=cc["City"],y=cc["Target"])

```


```python 
cc=pd.concat([cc,new],axis=1)
cc

```
![alt text](<Screenshot 2024-09-24 114618.png>)


```python 
import pandas as pd
from scipy import stats
import numpy as np

```



```python 
df=pd.read_csv('/content/Data_to_Transform.csv')
df

```
![alt text](<Screenshot 2024-10-01 084800.png>)


```python 
df.shape

```
![alt text](<Screenshot 2024-10-01 084827.png>)


```python 

df.skew()
```
![alt text](<Screenshot 2024-10-01 084851.png>)

```python 
np.log(df['Highly Positive Skew'])

```
![alt text](<Screenshot 2024-10-01 084859.png>)

```python 
np.reciprocal(df['Moderate Negative Skew'])

```

![alt text](<Screenshot 2024-10-01 084913.png>)

```python 
np.sqrt(df['Highly Positive Skew'])

```

![alt text](<Screenshot 2024-10-01 085406.png>)

```python 
df.skew()

```

![alt text](<Screenshot 2024-10-01 085415.png>)

```python 
df['Highly Positive Skew']=np.sqrt(df['Highly Positive Skew'])
df

```

![alt text](<Screenshot 2024-10-01 085428.png>)


```python 
df.skew()

```

![alt text](<Screenshot 2024-10-01 085449.png>)

```python 
df['Highly Positive Skew_boxcox'],parameters=stats.boxcox(df['Highly Positive Skew'])
df
```


![alt text](<Screenshot 2024-10-01 085504.png>)

```python 
df['Moderate Negative Skew_yeojohnson'],parameters=stats.yeojohnson(df['Moderate Negative Skew'])
df
```

![alt text](<Screenshot 2024-10-01 085534.png>)

```python 
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

```


```python 

sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```

![alt text](<Screenshot 2024-10-01 085550.png>)

```python 
sm.qqplot(np.reciprocal(df['Moderate Negative Skew']),line='45')
plt.show()

```
![alt text](<Screenshot 2024-10-01 085612.png>)

```python 
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

```


```python 
df['Moderate Negative Skew']=qt.fit_transform(df[['Moderate Negative Skew']])

```

```python 
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()

```
![alt text](<Screenshot 2024-10-01 085624.png>)

# RESULT:
   Thus Feature encodind and transformation process is performed on the given data.
       
