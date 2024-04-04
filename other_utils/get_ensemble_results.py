import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score
from sksurv.metrics import concordance_index_censored

# def top_10_each_group_box_plots(df):
#     cols = [
#         "BOR_PD",
#         "BOR_PR",
#         "BOR_SD",
#         "ECOGBL",
#         "AGE_BINARY",
#         "CSITES_LIVER",
#         "TNaive",
#     ]

#     clidf = pd.read_csv(
#         "../../outputs/csvs/labels/CLOG2210_with_bor_71_patients_multivar_stratified_DCR_PFS_OS_labels3.csv"
#     )
#     clidf = clidf.rename(columns={"PatID": "PatientID"})

#     df = df.sort_values("MissClassifications", ascending=False)

#     upper = df.head(20).merge(clidf, on="PatientID")
#     lower = df.tail(20).merge(clidf, on="PatientID")

#     for col in cols:
#         print(f"{col}: {ttest_ind(upper[col].values, lower[col].values)[1]}")

#     import pdb

#     pdb.set_trace()


if __name__ == "__main__":
    mainfolder = "../../../mlpipeline/outputs/RAPTOMICS/"

    ensemble_list = [
        # "rad/lassofeatsel_binpval_corr09_08-29-2023-07-58-22/feature_2/",
        # "rad/T2_cphfeatsel_08-28-2023-20-08-13/feature_2/",
        # "rad/T2_cphfeatsel_binpval_08-28-2023-20-22-09/feature_1/"
        # "path/lassofeatsel_nooutlier_stdonly_08-29-2023-07-34-24/feature_5/",
        # "rad/NEW_lassofeatsel_binpval_corr09_withtrainpred_10cv_300runs_08-30-2023-06-54-19/feature_3",
        "rad/heliyon_rebuttal/Stability08_corr08_Lassofeatsel_withtrainpred_10cv_300runs_01-28-2024-08-17-51/features_1/",
        # "path/NEW_cphfeatsel_nooutlier_stdonly_withtrainpred_10cv_300runs_08-30-2023-07-03-45/feature_2/",
        # "path/NEW_lassofeatsel_nooutlier_stdonly_withtrainpred_10cv_300runs_08-30-2023-07-08-06/feature_5/",
        "path/NEW_cphfeatsel_corr08_withtrainpred_10cv_300runs_08-30-2023-15-16-28/feature_4/",
    ]

    event_indicator = "Label"
    event_time = "Surv"

    for item in ensemble_list:
        files = (
            Path(mainfolder)
            .joinpath(ensemble_list[0])
            .joinpath("results")
            .glob("run*.csv")
        )

        cindices = []

        for fi in files:
            df = pd.read_csv(str(fi))
            # df = df[df.PatientID.str.contains("CLOG")]
            df = df.sort_values("PatientID")
            df["PatientID"] = df["PatientID"].astype(str)

            for i in range(1, len(ensemble_list)):
                _df = pd.read_csv(str(fi).replace(ensemble_list[0], ensemble_list[i]))
                # _df = _df[_df.PatientID.str.contains("CLOG")]
                _df = _df.sort_values("PatientID")
                _df["PatientID"] = _df["PatientID"].astype(str)

                df = df.merge(_df, on=["PatientID", "Label", "Surv"])

                # assert (df.PatientID.values == _df.PatientID.values).all()

                pred1 = df["Pred_x"].values.tolist()
                pred2 = df["Pred_y"].values.tolist()

                pred = [pred1[i] + pred2[i] for i in range(len(pred1))]
                df["Pred"] = pred

                # df["Pred"] = df["Pred"].values + _df["Pred"].values

            ci = concordance_index_censored(
                df[event_indicator].values.astype(bool),
                df[event_time].values,
                df["Pred"].values,
            )

            cindices.append(ci[0])

    print(
        f"Mean c-indices = {np.mean(cindices):.3f}, Std accuracy: {np.std(cindices):.2f} "
    )
