import sys 
from machineLearningUtilRadPath import MachineLearningPipeline
from machineLearningUtilRadPath import FS
import pandas as pd 

config = {} 
config['patientIdentifier'] = 'PatientID'
config['labelIdentifier'] = 'RisingPSALabel'
config['time2eventIdentifier'] = 'timeDiffDays'
config['phaseIdentifier'] = 'Phase'

no_features = [(2,5)]

featureset = "radpath"

for no_feature in no_features:
    for featselect in [FS.CPHELASTIC]:

        if featselect == FS.LASSO:
            fldrname = "LASSO"

        elif featselect == FS.RFE:
            fldrname = "LRRFE"

        elif featselect == FS.MI:
            fldrname = "MI"

        elif featselect == FS.CHI2:
            fldrname = "CHI2"

        elif featselect == FS.MRMR:
            fldrname = "MRMR"

        elif featselect == FS.CPH:
            fldrname = "CPH"

        elif featselect == FS.CPHELASTIC:
            fldrname = "CPHELASTIC"

        outputfoldername = f'{featureset}/{fldrname}_cca4_{no_feature}'

        mlp = MachineLearningPipeline(f'../outputs/features/{featureset}Features.csv',f'../outputs/labels/finalccfbcrlabelsJune2021.csv',outputfoldername,config,featselect)
        mlp.params['noFeatures'] = no_feature

        mlp.cross_validate()
        # mlp.cph_elastic_crossvalidate()
