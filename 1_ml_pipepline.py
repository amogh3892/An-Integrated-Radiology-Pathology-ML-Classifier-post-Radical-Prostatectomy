import sys
from machineLearningUtil import MachineLearningPipeline
from machineLearningUtil import FS
import pandas as pd

config = {}
config["patientIdentifier"] = "PatientID"
config["labelIdentifier"] = "RisingPSALabel"
config["time2eventIdentifier"] = "timeDiffDays"
config["phaseIdentifier"] = "Phase"

no_features = [2, 3, 4, 5]

featureset = sys.argv[1]

for no_feature in no_features:
    for featselect in [FS.CPHLASSO]:
        if featselect == FS.LASSO:
            fldrname = "LASSO"

        if featselect == FS.ELASTICNET:
            fldrname = "ELASTICNET"

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

        elif featselect == FS.CPHLASSO:
            fldrname = "CPHLASSO"

        elif featselect == FS.CPHRIDGE:
            fldrname = "CPHRIDGE"

        outputfoldername = f"{featureset}/{fldrname}_{no_feature}_NEW"

        mlp = MachineLearningPipeline(
            f"../outputs/features/{featureset}Features.csv",
            f"../outputs/labels/finalccfbcrlabelsJune2021.csv",
            outputfoldername,
            config,
            featselect,
        )
        mlp.params["noFeatures"] = no_feature

        mlp.cross_validate()
        # mlp.cph_elastic_crossvalidate()
