
import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

# read and get values in two numpy arrays
f = open('data.txt');
activity = []
features = []
for line in f:
	line = line.rstrip('\n')
	values = line.split(',');
	activity.append(values[0])
	features.append(values[1:])

np_features = np.array(features)
np_activity = np.array(activity)

# feature names - columns in pandas dataframe
feature_names = ['min_f', 'max_f', 'mean_f', 'std_dev_f', 'auto_f',
                     'fft1_f', 'fft2_f', 'fft3_f', 'peaks_f', 'valleys_f',
                     'peaks_std_f', 'valleys_std_f', 'air_f', 'ffta1f', 'ffta2f', 'ffta3f', 're_f', 'fe_f',
                     'min_b', 'max_b', 'mean_b', 'std_dev_b', 'auto_b',
                     'fft1_b', 'fft2_b', 'fft3_b', 'peaks_b', 'valleys_b',
                     'peaks_std_b', 'valleys_std_b', 'air_b', 'ffta1b', 'ffta2b', 'ffta3b', 're_b', 'fe_b',
                     'min_a', 'max_a', 'mean_a', 'std_dev_a', 'auto_a',
                     'fft1_a', 'fft2_a', 'fft3_a', 'peaks_a', 'valleys_a',
                     'peaks_std_a', 'valleys_std_a', 'air_a', 'ffta1a', 'ffta2a', 'ffta3a', 're_a', 'fe_a']

# create pandas dataframe
data_dict = {}
for i in range(len(feature_names)):
	data_dict[feature_names[i]] = np_features[:, i]
      
df = DataFrame(data_dict, columns=feature_names)
df['activity'] = np_activity
df['activity'] = pd.factorize(df['activity'])[0]

df['is_train'] = np.random.uniform(0, 1, len(df)) <= 1.0
train = df[df['is_train'] == True]#, df[df['is_train'] == False]
test = train

features = df.columns[:54]
y_train = train['activity']
y_test = y_train

# RFC
clf1 = RandomForestClassifier()
# fit model
clf1.fit(train[features], y_train)

avg_acc1 = []

preds1 = clf1.predict(test[features])

# calculate accuracy
correct_preds1 = 0.0
for i in range(len(y_test)):
	if (y_test[i] == preds1[i]):
		correct_preds1 = correct_preds1 + 1

accuracy1 = (correct_preds1 / len(y_test)) * 100
      
print accuracy1

filename1 = 'RFC_activity_model_new.sav'
joblib.dump(clf1, filename1)
