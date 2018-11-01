import numpy as np

from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score


from smiles_util import get_fingerprint

class RandomForestQSAR(object):
    def __init__(self, model_type='classifier', n_estimators=100):
        super(RandomForestQSAR,self).__init__()
        self.n_estimators = n_estimators
        self.model_type = model_type
        self.fitted = False
        
        if self.model_type == 'classifier':
            self.model = RFC(n_estimators=n_estimators)
        elif self.model_type == 'regressor':
            self.model = RFR(n_estimators=n_estimators)
        else:
            raise ValueError('invalid model type')
        return
		
    def load_model(self,path):
        self.model = joblib.load(path)
        return
		
    def save_model(self,path):
        joblib.dump(self.model,path)
        return
		
    def fit(self,smiles, labels,test_size = 0.3, random_state = 0):
        # Get fingerprints for SMILES (switch to encodings)
        fp = get_fingerprint(smiles)
        X_train, X_test, y_train, y_test = train_test_split(fp, labels, test_size = test_size, random_state =  random_state)
        # Fit the model using training data
        self.model.fit(X_train,y_train)
        # Predict on test set
        pred_test = self.model.predict(X_test)
        eval_metrics = []
        if self.model_type == 'classifier':
            # fpr: False Postive Rate
            # tpr: True Postive Rate
            fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_test)
            eval_metrics.append(metrics.auc(fpr, tpr))
            metrics_type = 'AUC'
        elif self.model_type == 'regressor':
            r2 = metrics.r2_score(y_test, pred_test)
            eval_metrics.append(r2)
            metrics_type = 'R^2 score'
            self.fitted = True
        return eval_metrics, metrics_type
		
    def predict(self,smiles):
        fps = get_fingerprint(smiles)
        assert len(smiles) == len(fps)

        clean_smiles = []
        clean_fps = []
        nan_smiles = []

        for i in range(len(fps)):
            if np.isnan(sum(fps[i])):
                nan_smiles.append(smiles[i])
            else:
                clean_smiles.append(smiles[i])
                clean_fps.append(fps[i])
                clean_fps = np.array(clean_fps)

        if self.fitted:
            if (len(clean_fps) > 0):
                prediction = self.model.predict(clean_fps)
            else:
                print('Warning: Nothing to predict all SMILES are invalid, please check input')
        else:
            print('Model has not been fitted yet')
            
        return clean_smiles, prediction, nan_smiles	