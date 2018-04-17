import numpy as np
import pandas as pd
from pandas import DataFrame
import sklearn
from sklearn.externals import joblib

def main(features_dict):
  
  feature_names = ['min_f', 'max_f', 'mean_f', 'std_dev_f', 'auto_f',
                     'fft1_f', 'fft2_f', 'fft3_f', 'peaks_f', 'valleys_f',
                     'peaks_std_f', 'valleys_std_f', 'air_f', 'ffta1f', 'ffta2f', 'ffta3f', 're_f', 'fe_f',
                     'min_b', 'max_b', 'mean_b', 'std_dev_b', 'auto_b',
                     'fft1_b', 'fft2_b', 'fft3_b', 'peaks_b', 'valleys_b',
                     'peaks_std_b', 'valleys_std_b', 'air_b', 'ffta1b', 'ffta2b', 'ffta3b', 're_b', 'fe_b',
                     'min_a', 'max_a', 'mean_a', 'std_dev_a', 'auto_a',
                     'fft1_a', 'fft2_a', 'fft3_a', 'peaks_a', 'valleys_a',
                     'peaks_std_a', 'valleys_std_a', 'air_a', 'ffta1a', 'ffta2a', 'ffta3a', 're_a', 'fe_a']

  activity_dict = {0:'walking', 1:'standing', 2:'standing_up', 3:'squatting', 4:'sitting_down', 5:'sitting', 6:'jumping'}

  model = joblib.load('RFC_activity_model_new.sav')

  df = DataFrame(features_dict, columns=feature_names)
  features = df.columns[:54]
  try:
    pred = model.predict(df[features])
    prob = model.predict_proba(df[features])[0]
  except:
    print df[features]

  print activity_dict[pred[0]],

  for i in range(len(prob)):
    if prob[pred[0]] == prob[i] and pred[0] != i:
        print '/', activity_dict[i],

  print prob[pred[0]]

  #print "[", 
  #for x in range(len(prob)):
  #  print activity_dict[x], prob[x],
  #print "]"

if __name__=='__main__':
    sys.exit(main(sys.argv[1]))
	
