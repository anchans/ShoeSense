data_Advait, data_Artur, data_Anchan contains data collected from us for each activity. Experiments folder in data_Advait and data_Artur contain the validation activity set experiments mentioned in the report.

Use of scripts:

flexServer.py - reads UDP packets sent from photon and saves the data into files

readnPlot.py - reads data from files and plots it

processing.py - reads raw data from files, computes features and saves to file

computeFeatures.py - computes features for real time data stream	

train_model.py - trains Random Forest Classifier using feature data in data.txt

activity_predictor.py - predicts activity from computed features and pre-trained model

realTimeClassify.py - gets real time data stream and send it through the machine learning pipeline
