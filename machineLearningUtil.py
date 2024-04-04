
import sys 
import numpy as np 
# from rpy2.robjects import DataFrame, FloatVector, IntVector, StrVector
# from rpy2.robjects.packages import importr
# from rpy2 import robjects as ro
# from rpy2.robjects import pandas2ri
# pandas2ri.activate()
from sklearn.metrics import roc_auc_score 
# rpackage_pROC = importr('pROC')
# rpackage_base = importr('base')
# rpackage_psych = importr('psych')
from scipy.stats import ttest_ind
import pandas as pd 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# from tools.data_processing import FeatureSelector
from fast_ml.feature_selection import get_constant_features
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.model_selection import StratifiedKFold,GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from dataUtil import DataUtil
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif, chi2
from sklearn.linear_model import Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.svm import SVC
from scipy.stats import ttest_ind
# import pymrmr
from enum import Enum
# from sklearn_pandas import DataFrameMapper
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
import random
from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import RandomSurvivalForest
import imblearn 
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt 

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)



class FS(Enum):
    MI = 1 
    RFE = 2 
    LASSO = 3 
    CHI2 = 4 
    MRMR = 5
    ELASTICNET = 6
    CPH = 7
    CPHELASTIC = 8
    CPHLASSO = 9
    CPHRIDGE = 10


class MachineLearningPipeline(object):
    def __init__(self,featuresPath,labelspath,outputfoldername,config,fs):

        """
        featuresPath: full path with extension 
                to feature set (supports xlsx, csv)

                Make sure there are no extra columns in the file other than 'patientIdentifier'

        labelspath: path to lables csv/xlsx. Should contain columns 'patientIdentifier', 
                'labelIdentifier', 'phaseIdentifier' as indicated in config dictionary 

                'phaseIdentifier' indicates whether the sample belongs to training split or test split

        outputfoldername: Name of the ouputfolder to store the results. 

        
        config: a config dictionary 
                patientIdentifier: (mandatory) column name which consists patient identifier
                labelIdentifier: (mandatory) column name which consists of label information
                phaseIdentifier: (mandatory) column name which consists of phase information

                to disable a particular setting in the pipleline, for ex (StandarNormalize),
                use self.disable([....]) function

               
        params: parameters based on the config 
                ex. if FeatSelectCorr is True, set consThreshold, corrThreshold or leave it default, 0.5, 0.8
                to update the parameters, use self.update([key:value]) function

        Use self.disable() function to disable the below settings
        StandardNormalize: disable standard mean and std normalization 
        MinMaxNormalize: disable min max normalization
        FeatSelectCorr: disable removal of correlated features 

        """

        self.config = config
        self.config['StandardNormalize'] = True 
        self.config['RobustNormalize'] = False
        self.config['MinMaxNormalize'] = True 
        self.config['FeatSelectCons'] = True 
        self.config['FeatSelectCorr'] = True 
        self.config['FeatSelectPvalue'] = True 
        self.config['FeatSelect'] = fs 
        self.config['SubSample'] = False
        self.config['OverSample'] = False

        self.params = {}
        self.params['consThreshold'] = 0.8 # if 1-x% of the features are constant, remove them. 
        self.params['corrThreshold'] = 0.9
        self.params['noFeatures'] = 4

        self.outputfolder =f"outputs/results/{outputfoldername}"
        DataUtil.mkdir(self.outputfolder) 

        if ('patientIdentifier' not in config) or ('labelIdentifier' not in config) or ('phaseIdentifier' not in config):
            print("patientIdentifier/labelIdentifier/phaseIdentifier not specified")
            sys.exit()

        self.patientIdentifier = config['patientIdentifier']
        self.labelIdentifier = config['labelIdentifier']
        self.phaseIdentifier = config['phaseIdentifier']
        self.time2eventidentifier = config['time2eventIdentifier']

        labelsdf = pd.read_csv(labelspath) if '.csv' in labelspath else pd.read_excel(labelspath)

        # train/ test case names 
        self.traincases = labelsdf[labelsdf[self.phaseIdentifier]=='train'][self.patientIdentifier].values.tolist()
        self.testcases = labelsdf[labelsdf[self.phaseIdentifier]=='test'][self.patientIdentifier].values.tolist()

        # train/ test labels 
        self.trainlabels = labelsdf[labelsdf[self.phaseIdentifier]=='train'][self.labelIdentifier].values.tolist()
        self.testlabels = labelsdf[labelsdf[self.phaseIdentifier]=='test'][self.labelIdentifier].values.tolist()

        # train/ test time2event data
        self.traintime2event = labelsdf[labelsdf[self.phaseIdentifier]=='train'][self.time2eventidentifier].values.tolist()
        self.testtime2event = labelsdf[labelsdf[self.phaseIdentifier]=='test'][self.time2eventidentifier].values.tolist()
        
        # train/ test features 
        feat_df = pd.read_csv(featuresPath) if '.csv' in featuresPath else pd.read_excel(featuresPath)
        feat_df = feat_df.dropna(axis=1)

        self.trainfeatures = None 
        train_feat_df = feat_df[feat_df[self.patientIdentifier].isin(self.traincases)]

        self.patientids = train_feat_df.pop(self.patientIdentifier)
        self.patientids = self.patientids.values.tolist()
        self.featnames = train_feat_df.columns

        self.orgtrainfeatures = train_feat_df 
        self.trainfeatures = train_feat_df

        self.orgtestfeatures = None
        self.testfeatures = None 

        if len(self.testcases)>0:
            test_feat_df = feat_df[feat_df[self.patientIdentifier].isin(self.testcases)]
            test_feat_df = test_feat_df.drop([self.patientIdentifier],axis=1)
            self.orgtestfeatures = test_feat_df 
            self.testfeatures = test_feat_df 


        # if (config['StandardNormalize'] is True) and (config['MinMaxNormalize'] is True):
        #     # mapper = make_pipeline(StandardScaler(),
        #     #                 MinMaxScaler())
        #     mapper = make_pipeline(RobustScaler())

        # elif (config['StandardNormalize'] is True):
        #     mapper = make_pipeline(StandardScaler())
        # elif (config['MinMaxNormalize'] is True):
        #     mapper = make_pipeline(StandardScaler())

        # self.strainfeatures = mapper.fit_transform(self.trainfeatures)
        # self.strainfeatures = pd.DataFrame(self.strainfeatures,columns=self.featnames)

        # # remove correlated features pipepline 
        # corrFs = self._get_correlated_features(self.strainfeatures,self.trainlabels,
        #             self.params['consThreshold'],self.params['corrThreshold'])

        # self.cstrainfeatures = corrFs.transform(self.strainfeatures)

        # self.csfeatnames = self.cstrainfeatures.columns

        # # self.cstrainfeatures.to_csv(f'outputs/features/radpathFeaturesPreprocessed.csv')


    def cross_validate(self):

        featcount = self._get_cross_validation_results(self.trainfeatures,self.trainlabels,self.traintime2event,10,300)
        DataUtil.writeJson(featcount,f'{self.outputfolder}/feat_count.json')

        
    # def cph_elastic_crossvalidate(self):
    #     config = self.config

    #     featnames = self.featnames 
        
    #     trainfeatures = self.trainfeatures
    #     trainlabels = self.trainlabels
    #     traintime2event = self.traintime2event

    #     trainlabelstime2event = np.array([(bool(trainlabels[i]),traintime2event[i]) for i in range(len(trainlabels))])

    #     X_train = pd.DataFrame(trainfeatures,columns=featnames)

    #     y_train = [(x[0],x[1]) for x in trainlabelstime2event]
    #     y_train = np.array(y_train,dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

    #     selectedfeatnames = featnames 



    #     # if config['StandardNormalize'] is True:
    #     #     X_train,X_test,stdscaler = self.apply_fit_transform(StandardScaler,X_train,None)

    #     # if config['RobustNormalize'] is True:
    #     #     X_train,X_test,stdscaler = self.apply_fit_transform(RobustScaler,X_train,None)

    #     # if config['MinMaxNormalize'] is True:
    #     #     X_train,X_test,stdscaler = self.apply_fit_transform(MinMaxScaler,X_train,None)


    #     if config['FeatSelectPvalue'] is True:
    #         X_train, X_test, selectedfeatnames = self._remove_by_pvalue(pd.DataFrame(X_train,columns=selectedfeatnames),
    #                     None,trainlabels)

    #     if config['FeatSelectCorr'] is True:
    #         X_train, X_test, selectedfeatnames = self._remove_correlated_features(pd.DataFrame(X_train,columns=selectedfeatnames),
    #                     None,y_train,
    #                     self.params['corrThreshold'])


    #     # if config['FeatSelectCons'] is True:
    #     #     X_train, X_test, selectedfeatnames = self._remove_constant_features(pd.DataFrame(X_train,columns=selectedfeatnames),
    #     #                 None,trainlabels,
    #     #                 self.params['consThreshold'])

    #     coxnet_pipe = make_pipeline(
    #         StandardScaler(),
    #         CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, max_iter=1000)
    #     )

    #     import warnings
    #     from sklearn.exceptions import ConvergenceWarning
    #     warnings.simplefilter("ignore", ConvergenceWarning)


    #     coxnet_pipe.fit(X_train, y_train)

    #     estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
    #     cv = KFold(n_splits=5, shuffle=True, random_state=0)
    #     gcv = GridSearchCV(
    #         make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9, max_iter=1000)),
    #         param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]},
    #         cv=cv,
    #         error_score=0.5,
    #         n_jobs=4).fit(X_train, y_train)

    #     cv_results = pd.DataFrame(gcv.cv_results_)


    #     alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
    #     mean = cv_results.mean_test_score
    #     std = cv_results.std_test_score

    #     fig, ax = plt.subplots(figsize=(9, 6))
    #     ax.plot(alphas, mean)
    #     ax.fill_between(alphas, mean - std, mean + std, alpha=.15)
    #     ax.set_xscale("log")
    #     ax.set_ylabel("concordance index")
    #     ax.set_xlabel("alpha")
    #     ax.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], c="C1")
    #     ax.axhline(0.5, color="grey", linestyle="--")
    #     ax.grid(True)

    #     plt.show()


    #     best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
    #     best_coefs = pd.DataFrame(
    #         best_model.coef_,
    #         index=selectedfeatnames,
    #         columns=["coefficient"]
    #     )

    #     non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
    #     print("Number of non-zero coefficients: {}".format(non_zero))

    #     non_zero_coefs = best_coefs.query("coefficient != 0")
    #     coef_order = non_zero_coefs.abs().sort_values("coefficient").index

    #     _, ax = plt.subplots(figsize=(6, 8))
    #     non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
    #     ax.set_xlabel("coefficient")
    #     ax.grid(True)


    #     plt.show()

    #     import pdb 
    #     pdb.set_trace()


    def _remove_constant_features(self,_trainfeaturesdf,_testfeaturesdf,_labels,_cThreshold):

        """
        _featuresdf: datafram of training features- preprocessed
        _labels: trainlabels
        _cTreshold: Threshold to remove constant features 
        """

        featnames = _trainfeaturesdf.columns
        
        removefeatures = get_constant_features(_trainfeaturesdf,threshold=_cThreshold)['Var'].tolist()

        selectedfeatnames = [x for x in featnames if x not in removefeatures]

        X_train = _trainfeaturesdf[selectedfeatnames].values

        if _testfeaturesdf is not None:
            X_test = _testfeaturesdf[selectedfeatnames].values
        else:
            X_test = None 

        return X_train, X_test, selectedfeatnames

    def _remove_correlated_features(self,_trainfeaturesdf,_testfeaturesdf,_labels,_rThreshold):

        """
        _featuresdf: datafram of training features- preprocessed
        _labels: trainlabels
        _rTreshold: Threshold to remove correlated features
        
        """

        featnames = _trainfeaturesdf.columns

        # Create correlation matrix
        corr_matrix = _trainfeaturesdf.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Find index of feature columns with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > _rThreshold)]

        selectedfeatnames = [x for x in featnames if x not in to_drop]

        X_train = _trainfeaturesdf[selectedfeatnames].values

        if _testfeaturesdf is not None:
            X_test = _testfeaturesdf[selectedfeatnames].values
        else:
            X_test = None 

        return X_train, X_test, selectedfeatnames


    # def _remove_correlated_features(self,_trainfeaturesdf,_testfeaturesdf,_labels,_cThreshold,_rThreshold):

    #     """
    #     _featuresdf: datafram of training features- preprocessed
    #     _labels: trainlabels
    #     _cTreshold: Threshold to remove constant features 
    #     _rTreshold: Threshold to remove correlated features
        
    #     """
    #     steps = [{'Constant Features': {'frac_constant_values': _cThreshold}},
    #                 {'Correlated Features': {'correlation_threshold': _rThreshold}}]


    #     fs = FeatureSelector()
    #     fs.fit(_trainfeaturesdf, _labels, steps)

    #     X_traindf = fs.transform(_trainfeaturesdf)
    #     selectedfeatnames = X_traindf.columns

    #     X_train = X_traindf.values
    #     X_test = fs.transform(_testfeaturesdf).values

    #     return X_train, X_test, selectedfeatnames



    def _remove_by_pvalue(self,_trainfeaturesdf,_testfeaturesdf,_labels):


        # _labels = np.array([x[0] for x in _labels])

        featnames = _trainfeaturesdf.columns

        pvalues = []

        # dct = {}

        for featname in featnames:
            pvalues.append(ttest_ind(_trainfeaturesdf[featname],_labels)[1])
            # dct[featname] = ttest_ind(_trainfeaturesdf[featname],_labels)[1]

        inds = np.where(np.array(pvalues) < 0.05)[0]

        selectecfeatnames = featnames[inds]

        X_train = _trainfeaturesdf[selectecfeatnames].values

        if _testfeaturesdf is not None:
            X_test = _testfeaturesdf[selectecfeatnames].values
        else:
            X_test = None 

        return X_train, X_test, selectecfeatnames



    def _smote_balance_training_set(self,X,y):
        columns = X.columns 
        sm = SMOTE(random_state=42)
        X_sm, y_sm = sm.fit_resample(X.values, y)
        X_sm = pd.DataFrame(X_sm,columns=columns)
        return X_sm, y_sm 


    def _get_best_parameters_RF(self,trainfeatures,trainlabels):

        estimator = RandomForestClassifier()

        param_grid = { 'n_estimators': [100,200],
                        'max_depth' : [3, 4, 5, 6, 7],
                        # 'min_samples_split': [0.01, 0.05, 0.10],
                        # 'min_samples_leaf': [0.01, 0.05, 0.10],
                        # 'criterion' :['gini', 'entropy']     ,
                        'criterion' :['gini']     ,
                        'n_jobs': [-1]
                        }

        gscv = GridSearchCV(estimator, param_grid, cv = 5,  n_jobs= -1, verbose = 1, scoring = 'roc_auc')

        gscv.fit(trainfeatures, trainlabels)

        best_params = gscv.best_params_
        best_score = gscv.best_score_

        print(best_params)


        return best_params


    def _get_best_parameters_LR(self,trainfeatures,trainlabels):

        estimator = LogisticRegression(max_iter=5000)


        space = {}

        # space['solver'] = ['liblinear','saga']
        # space['penalty'] = ['l1','l2']
        space['C'] = np.logspace(-3, 3, 7)

        gscv = GridSearchCV(estimator, space, cv = 5,  n_jobs= -1, verbose = 1, scoring = 'roc_auc')
        gscv.fit(trainfeatures, trainlabels)

        best_params = gscv.best_params_
        best_score = gscv.best_score_


        return best_params



    def _get_best_parameters_SVM(self,trainfeatures,trainlabels):

        estimator = SVC()
    
        param_grid = {'C': [0.1, 1, 10, 100, 1000], 
                    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                    'kernel': ['rbf','linear']} 

        gscv = GridSearchCV(estimator, param_grid, cv = 5,  n_jobs= -1, verbose = 1, scoring = 'roc_auc')

        gscv.fit(trainfeatures, trainlabels)

        best_params = gscv.best_params_
        best_score = gscv.best_score_

    
        return best_params




    def _RFE_FS(self,trainfeaturesdf,testfeaturesdf, trainlabels):


        featnames = trainfeaturesdf.columns
        # trainfeatures = trainfeaturesdf.values

        # estimator = RandomForestClassifier()

        # param_grid = { 'n_estimators': [200],
        #                 'max_depth' : [3, 4, 5, 6, 7],
        #                 'min_samples_split': [0.01, 0.05, 0.10],
        #                 'min_samples_leaf': [0.01, 0.05, 0.10],
        #                 'criterion' :['gini', 'entropy']     ,
        #                 'n_jobs': [-1]
        #                 }

        # gscv = GridSearchCV(estimator, param_grid, cv = 5,  n_jobs= -1, verbose = 1, scoring = 'roc_auc')

        # gscv.fit(trainfeaturesdf, trainlabels)

        # best_params = gscv.best_params_
        # best_score = gscv.best_score_


        # estimator.set_params(**best_params)

        estimator = LogisticRegression(C=100,max_iter=5000)

        # best_params = self._get_best_parameters_LR(trainfeaturesdf, trainlabels)
        # estimator.set_params(**best_params)

        selector = RFE(estimator,n_features_to_select=self.params['noFeatures'],verbose=50,step=1)

        selector.fit(trainfeaturesdf,trainlabels)

        selected = trainfeaturesdf.iloc[:,np.where(selector.ranking_==1)[0]]

        selected_featnames = featnames[np.where(selector.ranking_==1)[0]].values.tolist()

        X_train = trainfeaturesdf[selected_featnames].values 
        X_test = testfeaturesdf[selected_featnames].values 

        return X_train, X_test, selected_featnames 


    def _chi2_FS(self,trainfeaturesdf,testfeaturesdf, trainlabels):


        featnames = trainfeaturesdf.columns

        trainfeatures = trainfeaturesdf.values

        selector = SelectKBest(chi2, k=self.params['noFeatures'])

        selector.fit(trainfeatures, trainlabels)

        inds = selector.get_support(indices=True)

        selected_featnames = featnames[inds].values.tolist()

        X_train = trainfeaturesdf[selected_featnames].values 
        X_test = testfeaturesdf[selected_featnames].values 

        return X_train, X_test, selected_featnames 


    def _MI_FS(self,trainfeaturesdf,testfeaturesdf, trainlabels):


        featnames = trainfeaturesdf.columns

        trainfeatures = trainfeaturesdf.values

        selector = SelectKBest(mutual_info_classif, k=self.params['noFeatures'])

        selector.fit(trainfeatures, trainlabels)

        inds = selector.get_support(indices=True)

        selected_featnames = featnames[inds].values.tolist()

        X_train = trainfeaturesdf[selected_featnames].values 
        X_test = testfeaturesdf[selected_featnames].values 

        return X_train, X_test, selected_featnames 

    def _MRMR_FS(self,trainfeaturesdf,testfeaturesdf, trainlabels):

        trainfeaturesdf.insert (0, "Label", trainlabels)

        selected_featnames = pymrmr.mRMR(trainfeaturesdf, 'MID',self.params['noFeatures'])

        X_train = trainfeaturesdf[selected_featnames].values 
        X_test = testfeaturesdf[selected_featnames].values 

        return X_train, X_test, selected_featnames 

    def _Lasso_FS(self,trainfeaturesdf,testfeaturesdf, trainlabels):

        featnames = trainfeaturesdf.columns

        trainfeatures = trainfeaturesdf.values

        model = Lasso(alpha=0.005)

        # from yellowbrick.regressor import AlphaSelection

        # alphas = np.logspace(-10,-1,50)
        # model = LassoCV(alphas=alphas)

        # visualizer = None 
        # visualizer = AlphaSelection(model)
        # visualizer.fit(trainfeatures, trainlabels)
        # al = visualizer.estimator.alpha_
        # visualizer.show()

        # model = ElasticNet(alpha=0.005*2)
        # model = Lasso(alpha=al)

        selector = SelectFromModel(model, max_features=self.params['noFeatures'])
        # selector = SelectFromModel(model, threshold=0.001) #To pick nonzero weighted features

        selector.fit(trainfeatures, trainlabels)

        #selector.estimator_.coef_

        inds = selector.get_support(indices=True)


        # print(selector.estimator_.coef_)

        # import pdb 
        # pdb.set_trace()

        selected_featnames = featnames[inds].values.tolist()

        X_train = trainfeaturesdf[selected_featnames].values 
        X_test = testfeaturesdf[selected_featnames].values 

        return X_train, X_test, selected_featnames 


    def _ELASTICNET_FS(self,trainfeaturesdf,testfeaturesdf, trainlabels):


        featnames = trainfeaturesdf.columns

        trainfeatures = trainfeaturesdf.values

        # model = Lasso(alpha=0.01)

        # from yellowbrick.regressor import AlphaSelection

        # alphas = np.logspace(-10,-1,50)
        # model = LassoCV(alphas=alphas)

        # visualizer = None 
        # visualizer = AlphaSelection(model)
        # visualizer.fit(trainfeatures, trainlabels)
        # al = visualizer.estimator.alpha_
        # visualizer.show()


        model = ElasticNet(alpha=0.01,l1_ratio=0.9)

        selector = SelectFromModel(model, max_features=self.params['noFeatures'])
        # selector = SelectFromModel(model, threshold=0.001) #To pick nonzero weighted features

        selector.fit(trainfeatures, trainlabels)

        #selector.estimator_.coef_

        inds = selector.get_support(indices=True)


        # print(selector.estimator_.coef_)

        # import pdb 
        # pdb.set_trace()

        selected_featnames = featnames[inds].values.tolist()

        X_train = trainfeaturesdf[selected_featnames].values 
        X_test = testfeaturesdf[selected_featnames].values 

        return X_train, X_test, selected_featnames 

    def _CPH_FS(self,trainfeaturesdf,testfeaturesdf, trainlabelstime2event):

        def fit_and_score_features_cph(X, y):
            n_features = X.shape[1]
            scores = np.empty(n_features)
            m = CoxPHSurvivalAnalysis(alpha=0.00001)
            for j in range(n_features):
                Xj = X[:, j:j+1]
                m.fit(Xj, y)
                scores[j] = m.score(Xj, y)
            return scores

        trainlabelstime2event = np.array([(bool(x[0]),x[1]) for x in trainlabelstime2event],dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

        featnames = trainfeaturesdf.columns

        trainfeatures = trainfeaturesdf.values

        selector = SelectKBest(fit_and_score_features_cph, k=self.params['noFeatures'])

        selector.fit(trainfeatures, trainlabelstime2event)

        inds = selector.get_support(indices=True)

        selected_featnames = featnames[inds].values.tolist()

        X_train = trainfeaturesdf[selected_featnames].values 
        X_test = testfeaturesdf[selected_featnames].values 

        return X_train, X_test, selected_featnames 



    # def _CPHELASTIC_FS(self,trainfeaturesdf,testfeaturesdf, trainlabelstime2event):

    #     trainlabelstime2event = np.array([(bool(x[0]),x[1]) for x in trainlabelstime2event],dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

    #     featnames = trainfeaturesdf.columns

    #     trainfeatures = trainfeaturesdf.values


    #     # model = CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.5)
    #     # model = CoxPHSurvivalAnalysis(alpha=0.1)

    #     model = CoxnetSurvivalAnalysis(l1_ratio=1, alphas=np.array([0.04]))

    #     model.fit(trainfeatures,trainlabelstime2event)

    #     coefs = np.abs(model.coef_.ravel())

    #     sortedindices = np.argsort(coefs)[::-1]

    #     # selector = SelectFromModel(model, max_features=self.params['noFeatures'])
    #     # # selector = SelectFromModel(model, threshold=0.001) #To pick nonzero weighted features

    #     # selector.fit(trainfeatures, trainlabelstime2event)
    #     # #selector.estimator_.coef_

    #     # inds = selector.get_support(indices=True)

    #     inds = sortedindices[:self.params['noFeatures']]

    #     selected_featnames = featnames[inds].values.tolist()

    #     X_train = trainfeaturesdf[selected_featnames].values 
    #     X_test = testfeaturesdf[selected_featnames].values 

    #     return X_train, X_test, selected_featnames 


    def _CPHELASTIC_FS(self,trainfeaturesdf,testfeaturesdf, trainlabelstime2event):

        trainlabelstime2event = np.array([(bool(x[0]),x[1]) for x in trainlabelstime2event],dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

        featnames = trainfeaturesdf.columns

        trainfeatures = trainfeaturesdf.values

        model = CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01)
        model.fit(trainfeatures,trainlabelstime2event)

        coefsum = np.sum(model.coef_,axis=1)
        coefsum = np.abs(coefsum)

        sortedindices = np.argsort(coefsum)[::-1]

        inds = sortedindices[:self.params['noFeatures']]

        selected_featnames = featnames[inds].values.tolist()

        X_train = trainfeaturesdf[selected_featnames].values 
        X_test = testfeaturesdf[selected_featnames].values 


        # model = CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01)
        # model.fit(trainfeatures,trainlabelstime2event)



        # coxnet_pipe = make_pipeline(
        #     CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, max_iter=10000)
        # )
        # coxnet_pipe.fit(trainfeatures, trainlabelstime2event)

        # estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
        # cv = KFold(n_splits=3, shuffle=True, random_state=0)
        # gcv = GridSearchCV(
        #     make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9, max_iter=10000)),
        #     param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]},
        #     cv=cv,
        #     error_score=0.5,
        #     n_jobs=4).fit(trainfeatures, trainlabelstime2event)


        # cv_results = pd.DataFrame(gcv.cv_results_)

        # alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
        # mean = cv_results.mean_test_score
        # std = cv_results.std_test_score

        # fig, ax = plt.subplots(figsize=(9, 6))
        # ax.plot(alphas, mean)
        # ax.fill_between(alphas, mean - std, mean + std, alpha=.15)
        # ax.set_xscale("log")
        # ax.set_ylabel("concordance index")
        # ax.set_xlabel("alpha")
        # ax.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], c="C1")
        # ax.axhline(0.5, color="grey", linestyle="--")
        # ax.grid(True)

        # plt.show()

        # import pdb 
        # pdb.set_trace()


        # best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
        # best_coefs = pd.DataFrame(
        #     best_model.coef_,
        #     index=featnames,
        #     columns=["coefficient"]
        # )

        # non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
        # print("Number of non-zero coefficients: {}".format(non_zero))

        # non_zero_coefs = best_coefs.query("coefficient != 0")
        # coef_order = non_zero_coefs.abs().sort_values("coefficient").index

        # selected_featnames = coef_order[:self.params['noFeatures']].tolist()

        # X_train = trainfeaturesdf[selected_featnames].values 
        # X_test = testfeaturesdf[selected_featnames].values 

        return X_train, X_test, selected_featnames 


    def _CPHLASSO_FS(self,trainfeaturesdf,testfeaturesdf, trainlabelstime2event):

        trainlabelstime2event = np.array([(bool(x[0]),x[1]) for x in trainlabelstime2event],dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

        featnames = trainfeaturesdf.columns

        trainfeatures = trainfeaturesdf.values

        model = CoxnetSurvivalAnalysis(l1_ratio=1, alpha_min_ratio=0.01)
        model.fit(trainfeatures,trainlabelstime2event)

        coefsum = np.sum(model.coef_,axis=1)
        coefsum = np.abs(coefsum)

        sortedindices = np.argsort(coefsum)[::-1]

        inds = sortedindices[:self.params['noFeatures']]

        selected_featnames = featnames[inds].values.tolist()

        X_train = trainfeaturesdf[selected_featnames].values 
        X_test = testfeaturesdf[selected_featnames].values 

        return X_train, X_test, selected_featnames 

    def _CPHRIDGE_FS(self,trainfeaturesdf,testfeaturesdf, trainlabelstime2event):

        trainlabelstime2event = np.array([(bool(x[0]),x[1]) for x in trainlabelstime2event],dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

        featnames = trainfeaturesdf.columns

        trainfeatures = trainfeaturesdf.values

        model = CoxnetSurvivalAnalysis(l1_ratio=0.85, alpha_min_ratio=0.01)
        model.fit(trainfeatures,trainlabelstime2event)

        coefsum = np.sum(model.coef_,axis=1)
        coefsum = np.abs(coefsum)

        sortedindices = np.argsort(coefsum)[::-1]

        inds = sortedindices[:self.params['noFeatures']]

        selected_featnames = featnames[inds].values.tolist()

        X_train = trainfeaturesdf[selected_featnames].values 
        X_test = testfeaturesdf[selected_featnames].values 

        return X_train, X_test, selected_featnames 


    def _cv_RFE_FS(self,trainfeaturesdf,trainlabels,runs = 10):

        trainfeatures = trainfeaturesdf.values 
        trainfeatnames = trainfeaturesdf.columns.values.tolist() 

        trainlabels = np.array(trainlabels)

        featcount = dict.fromkeys(trainfeatnames,0)

        for run in range(runs):
            skf = StratifiedKFold(n_splits=2,shuffle=True)
            skf.get_n_splits(trainfeatures, trainlabels)

            for train_index, test_index in skf.split(trainfeatures, trainlabels):

                X_train, X_test = trainfeatures[train_index], trainfeatures[test_index]
                y_train, y_test = trainlabels[train_index], trainlabels[test_index]

                selected_featnames = self._RFE_FS(pd.DataFrame(X_train,columns=trainfeatnames),y_train)

                for sf in selected_featnames:
                    featcount[sf] = featcount[sf] + 1

        return featcount


    def apply_fit_transform(self,transClass,trainfeatures, testfeatures=None):
        obj = transClass()
        trainfeatures = obj.fit_transform(trainfeatures)

        if testfeatures is not None:
            testfeatures = obj.transform(testfeatures)
        return trainfeatures,testfeatures,obj


    def subsample_indices(self,y, size=None):

        if size == None:
            size = np.min([(y==0).sum(),(y==1).sum()])

        indices = {}
        target_values = set(y)
        for t in target_values:
            indices[t] = [i for i in range(len(y)) if y[i] == t]
        min_len = min(size, min([len(indices[t]) for t in indices]))
        for t in indices:
            if len(indices[t]) > min_len:
                indices[t] = random.sample(indices[t], min_len)


        inds = []
        inds.extend(indices[0])
        inds.extend(indices[1])

        return inds

    def random_over_sample(self,trainfeatures,trainlabels):

        time2event = trainlabels[:,1]
        trainlabels = trainlabels[:,0]

        trainfeatures = np.column_stack((trainfeatures,time2event))

        oversample = RandomOverSampler(sampling_strategy='minority')

        X_over, y_over = oversample.fit_resample(trainfeatures, trainlabels)
        
        time2event = X_over[:,-1]
        X_over = X_over[:,:-1]

        y_over = np.array([ (y_over[i],time2event[i]) for i in range(len(y_over))])

        return X_over, y_over


    def _get_cross_validation_results(self,trainfeaturesdf,trainlabels,traintime2event,cv,runs):

        config = self.config

        featnames = self.featnames 
         

        outputfolder = self.outputfolder

        traincolumns = trainfeaturesdf.columns

        trainfeatures = trainfeaturesdf.values
        trainlabels = np.array(trainlabels)
        traintime2event = np.array(traintime2event)

        trainlabelstime2event = np.array([(bool(trainlabels[i]),traintime2event[i]) for i in range(len(trainlabels))])

        patientids = np.array(self.patientids)

        conis = []
        conis_overall = [] 

        featcount = dict.fromkeys(featnames,0)

        for run in range(runs):
            conirun = [] 

            skf = StratifiedKFold(n_splits=cv,shuffle=True,random_state=run*4)
            skf.get_n_splits(trainfeatures, trainlabels)

            preds = None 

            count = 0 
            for train_index, test_index in skf.split(trainfeatures, trainlabels):
                selectedfeatnames = self.featnames

                X_train, X_test = trainfeatures[train_index], trainfeatures[test_index]
                y_train, y_test = trainlabelstime2event[train_index], trainlabelstime2event[test_index]

                testpatientids = patientids[test_index]


                # remove columns with nan value 


                if config['StandardNormalize'] is True:
                    X_train,X_test,stdscaler = self.apply_fit_transform(StandardScaler,X_train,X_test)

                if config['RobustNormalize'] is True:
                    X_train,X_test,stdscaler = self.apply_fit_transform(RobustScaler,X_train,X_test)

                if config['MinMaxNormalize'] is True:
                    X_train,X_test,stdscaler = self.apply_fit_transform(MinMaxScaler,X_train,X_test)


                if config['FeatSelectPvalue'] is True:
                    X_train, X_test, selectedfeatnames = self._remove_by_pvalue(pd.DataFrame(X_train,columns=selectedfeatnames),
                                pd.DataFrame(X_test,columns=selectedfeatnames),np.array([x[0] for x in y_train]))

                if config['FeatSelectCons'] is True:
                    X_train, X_test, selectedfeatnames = self._remove_constant_features(pd.DataFrame(X_train,columns=selectedfeatnames),
                                pd.DataFrame(X_test,columns=selectedfeatnames),np.array([x[0] for x in y_train]),
                                self.params['consThreshold'])


                if config['FeatSelectCorr'] is True:
                    X_train, X_test, selectedfeatnames = self._remove_correlated_features(pd.DataFrame(X_train,columns=selectedfeatnames),
                                pd.DataFrame(X_test,columns=selectedfeatnames),np.array([x[0] for x in y_train]),
                                self.params['corrThreshold'])

                if self.config['OverSample']:
                    X_train,y_train = self.random_over_sample(X_train,y_train)

                elif self.config['SubSample']:
                    y_train_labels = np.array([x[0] for x in y_train])
                    y_train_inds = self.subsample_indices(y_train_labels,size=None)
                    X_train = X_train[y_train_inds]
                    y_train = y_train[y_train_inds]

                if config['FeatSelect'] == FS.MRMR:
                    X_train, X_test, selectedfeatnames = self._MRMR_FS(pd.DataFrame(X_train,columns=selectedfeatnames),
                                pd.DataFrame(X_test,columns=selectedfeatnames),np.array([x[0] for x in y_train]))

                if config['FeatSelect'] == FS.MI:
                    X_train, X_test, selectedfeatnames = self._MI_FS(pd.DataFrame(X_train,columns=selectedfeatnames),
                                pd.DataFrame(X_test,columns=selectedfeatnames),np.array([x[0] for x in y_train]))


                if config['FeatSelect'] == FS.LASSO:
                    X_train, X_test, selectedfeatnames = self._Lasso_FS(pd.DataFrame(X_train,columns=selectedfeatnames),
                                pd.DataFrame(X_test,columns=selectedfeatnames),np.array([x[0] for x in y_train]))

                if config['FeatSelect'] == FS.CHI2:
                    X_train, X_test, selectedfeatnames = self._chi2_FS(pd.DataFrame(X_train,columns=selectedfeatnames),
                                pd.DataFrame(X_test,columns=selectedfeatnames),y_train)

                if config['FeatSelect'] == FS.RFE:
                    X_train, X_test, selectedfeatnames = self._RFE_FS(pd.DataFrame(X_train,columns=selectedfeatnames),
                                pd.DataFrame(X_test,columns=selectedfeatnames),np.array([x[0] for x in y_train]))


                if config['FeatSelect'] == FS.ELASTICNET:
                    X_train, X_test, selectedfeatnames = self._ELASTICNET_FS(pd.DataFrame(X_train,columns=selectedfeatnames),
                                pd.DataFrame(X_test,columns=selectedfeatnames),np.array([x[0] for x in y_train]))

                if config['FeatSelect'] == FS.CPH:
                    X_train, X_test, selectedfeatnames = self._CPH_FS(pd.DataFrame(X_train,columns=selectedfeatnames),
                                pd.DataFrame(X_test,columns=selectedfeatnames),y_train)


                if config['FeatSelect'] == FS.CPHELASTIC:
                    X_train, X_test, selectedfeatnames = self._CPHELASTIC_FS(pd.DataFrame(X_train,columns=selectedfeatnames),
                                pd.DataFrame(X_test,columns=selectedfeatnames),y_train)

                if config['FeatSelect'] == FS.CPHLASSO:
                    X_train, X_test, selectedfeatnames = self._CPHLASSO_FS(pd.DataFrame(X_train,columns=selectedfeatnames),
                                pd.DataFrame(X_test,columns=selectedfeatnames),y_train)

                if config['FeatSelect'] == FS.CPHRIDGE:
                    X_train, X_test, selectedfeatnames = self._CPHRIDGE_FS(pd.DataFrame(X_train,columns=selectedfeatnames),
                                pd.DataFrame(X_test,columns=selectedfeatnames),y_train)
                                
                # selected_featnames = self._MRMR_FS(pd.DataFrame(X_train,columns=traincolumns),y_train)
                # selected_featnames = ['Sub-Graph: Skewness edge length', 'Arch: Area of polygons', 'Shape: Mean Fourier  5', 'Shape: Median Fourier  5', r'Shape: 5% / 95% smoothness', r'Shape: 5% / 95% invariant  1', r'Shape: 5% / 95% Fourier  8']

                for sf in selectedfeatnames:
                    featcount[sf] = featcount[sf] + 1

                # inds = [featnames.index(x) for x in selected_featnames]

                # X_train = X_train[:,inds]
                # X_test = X_test[:,inds]

                y_train = [(x[0],x[1]) for x in y_train]
                y_train = np.array(y_train,dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

                y_test_cens = [bool(x[0]) for x in y_test]
                y_test_surv = [x[1] for x in y_test]

                y_test = [(x[0],x[1]) for x in y_test]
                y_test = np.array(y_test,dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

                clf = CoxPHSurvivalAnalysis(alpha=0.001)
                # clf = RandomSurvivalForest(n_estimators=1000,
                #            min_samples_split=6,
                #            min_samples_leaf=3,
                #            max_features="sqrt",
                #            n_jobs=-1,
                #            random_state=42)

                # clf = RandomSurvivalForest(n_estimators=1000)

                try:
                    clf.fit(X_train,y_train)
                except:
                    import pdb 
                    pdb.set_trace()

                pred = clf.predict(X_test)

                try:
                    result = concordance_index_censored(y_test_cens,y_test_surv, pred)
                    coni = result[0]
                except:
                    coni = 0


                _fold = [count]*len(y_test_cens)

                _pred = np.column_stack((testpatientids,pred,_fold,y_test_cens,y_test_surv))

                preds = _pred if preds is None else np.vstack((preds,_pred))
                
                conirun.append(coni)

                count = count + 1
                print(f'************************************ Concordance Index = {coni} ***************************************')


                

            predictionsdf = pd.DataFrame(preds,columns=['PatientID','Pred','Fold','Label','Surv'])
            predictionsdf['Label'] = predictionsdf['Label'].astype(bool)
            result_overall = concordance_index_censored(predictionsdf['Label'],predictionsdf['Surv'], predictionsdf['Pred'])

            predictionsdf.to_csv(f'{outputfolder}/run_{run}_predictions.csv',index=None)

            conis.append(tuple(conirun))
            # conis = np.array(conirun) if conis is None else np.vstack((conis,np.array(conirun)))
            conis_overall.append((f'run_{run}',result_overall[0]))


        conis = np.array(conis).T

        pd.DataFrame(conis,columns=[f'run_{i}' for i in range(runs)]).to_csv(f'{outputfolder}/performance.csv',index=None)
        overall_conis_df = pd.DataFrame(conis_overall,columns=['Run','ConcordanceIndex'])

        overall_conis_df = overall_conis_df.append({'Run': 'Mean','ConcordanceIndex': np.mean(overall_conis_df['ConcordanceIndex'].values)}, ignore_index=True)
        overall_conis_df = overall_conis_df.append({'Run': 'Std','ConcordanceIndex': np.std(overall_conis_df['ConcordanceIndex'].values)}, ignore_index=True)


        overall_conis_df.to_csv(f'{outputfolder}/performanceOverall.csv',index=None)

        
        return featcount

# class PerformanceMetrics(object):
#     def __init__(self,ytrue,ypred,cutoff=None,cutoffvariable="threshold"):

#         self.ytrue = ytrue 
#         self.ypred = ypred 

#         self.aucobj = self._get_auc_obj() 
#         self.auc = self.aucobj[8][0]
#         self.ci = rpackage_pROC.ci(self.aucobj, x="best")
#         self.lowci = self.ci[0]
#         self.highci = self.ci[2]
#         self.binary = None 


#         if cutoff is None:
#             cutoffmetrics = rpackage_pROC.coords(self.aucobj,"best", cutoffvariable,ret=StrVector(["threshold","specificity", "sensitivity","accuracy","ppv","npv"]))
#             # cutoffmetrics = rpackage_cutpointr(self.aucobj,"best", cutoffvariable,ret=StrVector(["threshold","specificity", "sensitivity","accuracy","ppv","npv"]))

#         else:
#             cutoffmetrics = rpackage_pROC.coords(self.aucobj,cutoff, cutoffvariable,ret=StrVector(["threshold","specificity", "sensitivity","accuracy","ppv","npv"]))

    
#             if cutoffvariable == "threshold":
#                 binary_pred = [1 if x > cutoff else 0 for x in ypred]
#                 self.binary = binary_pred


#         self.threshold = cutoffmetrics["threshold"][0]
#         self.specificity = cutoffmetrics["specificity"][0]
#         self.sensitivity = cutoffmetrics["sensitivity"][0]
#         self.accuracy = cutoffmetrics["accuracy"][0]
#         self.ppv = cutoffmetrics["ppv"][0]
#         self.npv = cutoffmetrics["npv"][0]


#     def _get_auc_obj(self):
#         ytrue = self.ytrue 
#         ypred = self.ypred

#         ytrue = np.concatenate((ytrue, ytrue), axis=0)
#         ypred = np.concatenate((ypred, ypred), axis=0)

#         rpackage_pROC = importr('pROC')
#         rpackage_base = importr('base')
#         rpackage_psych = importr('psych')

#         #  2000 stratified bootstrap replicates
#         r_rocobj = rpackage_pROC.roc(ytrue, ypred)

#         return r_rocobj

#     def compare_auc(self,obj):

#         t1 = rpackage_pROC.roc_test(self.aucobj,obj.aucobj)

#         pvalue = t1[6][0]

#         try:
#             pvalue = float(pvalue)
#         except:
#             pvalue = t1[7][0]


#         return pvalue


    
#     @property
#     def pvalue(self):
#         ypred = self.ypred
#         nullhypo = [0.5]*len(ypred)

#         s,p = ttest_ind(ypred,nullhypo)

#         return p 




if __name__ == "__main__":

    # ytrue = [1,0,1,0,1,0,1,0,1,0,1,0]
    # ypred = [0.9,0.2,0.8,0.3,0.7,0.1,0.88,0.4,1,0.3,0.99,0]

    # pm = PerformanceMetrics(ytrue,ypred)


    # print(pm.auc, pm.ci, pm.accuracy, pm.sensitivity, pm.specificity)

    config = {} 
    config['patientIdentifier'] = 'PatientID'
    config['labelIdentifier'] = 'Label'
    config['phaseIdentifier'] = 'Phase'

    mlp = MachineLearningPipeline(f'outputs/features/radFeatures.csv',f'outputs/labels/finalccfbcrlabels.csv',config)

    import pdb 
    pdb.set_trace()
