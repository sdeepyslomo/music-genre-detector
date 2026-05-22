import os
import librosa    
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import kagglehub
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


DATASET_PATH = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
print("Path to dataset files:", DATASET_PATH)
print("Inside:", os.listdir(DATASET_PATH))

DATASET_PATH = os.path.join(DATASET_PATH, "Data", "genres_original")

features = []
labels = []

for genre in os.listdir(DATASET_PATH):
    genre_path = os.path.join(DATASET_PATH, genre)

    if not os.path.isdir(genre_path):
        continue 
    
    for file in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file)

        try:
            audio, sample_rate = librosa.load(file_path)
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            mfccs_mean = np.mean(mfccs,axis=1)
            mfccs_std = np.std(mfccs,axis=1)
            delta = librosa.feature.delta(mfccs)
            delta_mean = np.mean(delta, axis=1)
            delta_std = np.std(delta, axis=1)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio,sr=sample_rate)
            centroid_mean = np.mean(spectral_centroid)
            centroid_std = np.std(spectral_centroid)
            chroma = librosa.feature.chroma_stft(y=audio, sr= sample_rate)
            chroma_mean = np.mean(chroma, axis=1)
            chroma_std = np.std(chroma, axis=1)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio,sr=sample_rate)
            bandwidth_mean = np.mean(spectral_bandwidth)
            bandwidth_std = np.std(spectral_bandwidth)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio,sr=sample_rate)
            contrast_mean = np.mean(spectral_contrast,axis=1)
            contrast_std = np.std(spectral_contrast,axis=1)
            rms = librosa.feature.rms(y=audio)
            rms_mean = np.mean(rms)
            rms_std = np.std(rms)
            tempo = float(librosa.feature.tempo(y=audio,sr=sample_rate)[0])
            spectral_flatness = librosa.feature.spectral_flatness(y=audio)
            flatness_mean = np.mean(spectral_flatness)
            flatness_std = np.std(spectral_flatness)
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sample_rate)
            tonnetz_mean = np.mean(tonnetz, axis=1)
            tonnetz_std = np.std(tonnetz, axis=1)
            conc = np.concatenate((mfccs_mean
                                   , mfccs_std
                                   , delta_mean
                                   , delta_std
                                   , chroma_mean
                                   , chroma_std
                                   ,[centroid_mean]
                                   ,[centroid_std]
                                   ,[bandwidth_mean]
                                   ,[bandwidth_std]
                                   ,contrast_mean
                                   ,contrast_std
                                   ,[rms_mean]
                                   ,[rms_std]
                                   ,[tempo]
                                   ,[flatness_mean]
                                   ,[flatness_std]
                                   ,tonnetz_mean
                                   ,tonnetz_std))
            features.append(conc)
            labels.append(genre)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

print("Total samples:", len(features))     


x = np.array(features)
y = np.array(labels)

#encoder to convert ['Jazz','Metal',...] into [0,1,...]
encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("x shape:", x.shape)
print("y shape:", y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,test_size=0.2, random_state=42
)
print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))

#pipeline and gridsearch for svc
pipeline_svc= Pipeline([('scaler',StandardScaler())
                     ,('selector', SelectKBest(score_func=f_classif, k=90))
                     ,('svc',SVC())])
param_grid_svc= {
    'svc__C': [1, 10, 80, 100, 1000],
    'svc__gamma': [0.1, 0.01, 0.08, 0.001, 0.0001],
    'svc__kernel': ['rbf']
}
svc_grid = GridSearchCV(pipeline_svc, param_grid_svc, cv=5, verbose=2)
svc_grid.fit(x_train, y_train)
print("SVC Accuracy:", accuracy_score(y_test,svc_grid.predict(x_test)))

#pipeline and gridsearch for rf
pipeline_rf = Pipeline([('selector', SelectKBest(score_func=f_classif,k=90))
                       ,('rf',RandomForestClassifier())]) 
param_grid_rf = {
    'rf__n_estimators': [100,200],
    'rf__max_depth': [10,20,None],
    'rf__min_samples_split': [2,5,10],
    'rf__min_samples_leaf': [1,2,4]
}
rf_grid = GridSearchCV(pipeline_rf, param_grid_rf, cv=5, verbose=2)
rf_grid.fit(x_train,y_train)
print("RF Accuracy:", accuracy_score(y_test, rf_grid.predict(x_test)))

#pipeline and gridsearch for xgb
pipeline_xgb = Pipeline([('selector', SelectKBest(score_func=f_classif,k=90)),('xg', XGBClassifier())])
param_grid_xgb = {
    'xg__n_estimators': [100,200],
    'xg__max_depth': [3,5,7],
    'xg__learning_rate':[0.01,0.1],
    'xg__subsample': [0.8,1.0]
}
xgb_grid = GridSearchCV(pipeline_xgb,param_grid_xgb,cv=5,verbose=2)
xgb_grid.fit(x_train,y_train)
print("XGB Accuracy: ", accuracy_score(y_test, xgb_grid.predict(x_test)))


file_path = "data/silvera.wav"
audio, sample_rate = librosa.load(file_path)
mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
mfccs_mean = np.mean(mfccs,axis=1)
mfccs_std = np.std(mfccs,axis=1)
delta = librosa.feature.delta(mfccs)
delta_mean = np.mean(delta, axis=1)
delta_std = np.std(delta, axis=1)
spectral_centroid = librosa.feature.spectral_centroid(y=audio,sr=sample_rate)
centroid_mean = np.mean(spectral_centroid)
centroid_std = np.std(spectral_centroid)
chroma = librosa.feature.chroma_stft(y=audio,sr=sample_rate)
chroma_mean = np.mean(chroma, axis=1)
chroma_std = np.std(chroma, axis=1)
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio,sr=sample_rate)
bandwidth_mean = np.mean(spectral_bandwidth)
bandwidth_std = np.std(spectral_bandwidth)
spectral_contrast = librosa.feature.spectral_contrast(y=audio,sr=sample_rate)
contrast_mean = np.mean(spectral_contrast,axis=1)
contrast_std = np.std(spectral_contrast,axis=1)
rms = librosa.feature.rms(y=audio)
rms_mean = np.mean(rms)
rms_std = np.std(rms)
tempo = float(librosa.feature.tempo(y=audio,sr=sample_rate)[0])
spectral_flatness = librosa.feature.spectral_flatness(y=audio)
flatness_mean = np.mean(spectral_flatness)
flatness_std = np.std(spectral_flatness)
tonnetz = librosa.feature.tonnetz(y=audio, sr=sample_rate)
tonnetz_mean = np.mean(tonnetz, axis=1)
tonnetz_std = np.std(tonnetz, axis=1)
conc = np.concatenate((mfccs_mean
                       , mfccs_std
                       , delta_mean
                       , delta_std
                       , chroma_mean
                       ,chroma_std
                       ,[centroid_mean]
                       ,[centroid_std]
                       ,[bandwidth_mean]
                       ,[bandwidth_std]
                       ,contrast_mean
                       ,contrast_std
                       ,[rms_mean]
                       ,[rms_std]
                       ,[tempo]
                       ,[flatness_mean]
                       ,[flatness_std]
                       ,tonnetz_mean
                       ,tonnetz_std))
conc = conc.reshape(1,-1)

#for prediction
svc_pred = svc_grid.best_estimator_.predict(conc)
print("Predicted genre:", encoder.inverse_transform(svc_pred)[0])
print("Best parameter:", svc_grid.best_params_)

rf_pred = rf_grid.best_estimator_.predict(conc)
print("Predicted genre:", encoder.inverse_transform(rf_pred)[0])
print("Best parameter:", rf_grid.best_params_)

xgb_pred = xgb_grid.best_estimator_.predict(conc)
print("Predicted genre:", encoder.inverse_transform(xgb_pred)[0])
print("Best parameter:", xgb_grid.best_params_)

print("Feature length:", x.shape[1])

#confusion matrix & classification report of svc
conf_matx_svc = confusion_matrix(y_test, svc_grid.best_estimator_.predict(x_test))
print(conf_matx_svc)
print(classification_report(y_test,svc_grid.best_estimator_.predict(x_test)))

#confusion matrix & classification of rf
conf_matx_rf = confusion_matrix(y_test, rf_grid.best_estimator_.predict(x_test))
print(conf_matx_rf)
print(classification_report(y_test,rf_grid.best_estimator_.predict(x_test)))

#confusion matrix and classification of xgb
conf_matx_xgb = confusion_matrix(y_test, xgb_grid.best_estimator_.predict(x_test))
print(conf_matx_xgb)
print(classification_report(y_test
                            , xgb_grid.best_estimator_.predict(x_test)
                            ,target_names=encoder.classes_))

print(encoder.classes_)


pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)
genres = np.unique(y)
for genre in genres:
    indices = y == genre
    plt.scatter(
        x_pca[indices, 0],
        x_pca[indices, 1],
        label=genre
    )
plt.legend()
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Music Genres")
plt.show()









    

