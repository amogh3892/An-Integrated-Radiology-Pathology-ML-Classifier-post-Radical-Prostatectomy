import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score
from sksurv.metrics import concordance_index_censored
from lifelines import CoxPHFitter, KaplanMeierFitter, plotting

plt.rcParams.update({"font.size": 30})


def plt_km_curve(clnDFKM, thresh, predcol="PredB"):
    # --------create Kaplan-Meier curves for each group--------

    outcome = "Surv"
    label = "Label"

    plt.figure(figsize=(15, 8))

    low_risk_label = clnDFKM[predcol] <= thresh
    kmf_low_risk = KaplanMeierFitter()
    kmf_low_risk.fit(
        clnDFKM[outcome][low_risk_label],
        clnDFKM[label][low_risk_label],
        label="Low risk",
    )
    kmf_low_risk.plot(ci_show=True, ci_legend=False, color="blue")

    high_risk_label = clnDFKM[predcol] > thresh
    kmf_high_risk = KaplanMeierFitter()
    kmf_high_risk.fit(
        clnDFKM[outcome][high_risk_label],
        clnDFKM[label][high_risk_label],
        label="High risk",
    )

    ax = kmf_high_risk.plot(ci_show=True, ci_legend=False, color="red")
    plotting.add_at_risk_counts(
        kmf_low_risk,
        kmf_high_risk,
        rows_to_show=["At risk"],
        ypos=-0.2,
        fontsize=22,
    )

    plt.show()

    # # OS
    # plt.tight_layout()
    # ax.legend(loc="best", fontsize="13")
    # ax.set_xlabel("")
    # plt.text(
    #     18,
    #     0,
    #     # "HR = 4.03 [95% CI :  1.30-12.50], p=0.016",
    #     "HR = 4.29 [95% CI : 1.23 - 14.98], p = 0.022",
    #     ha="center",
    #     va="center",
    #     fontsize=22,
    # )

    # # PFS
    # plt.tight_layout()
    # ax.legend(loc="best", fontsize="13")
    # ax.set_xlabel("")
    # plt.text(
    #     6,
    #     0,
    #     "HR = 4.03 [95% CI :  1.30-12.50], p=0.016",
    #     # "HR = 4.29 [95% CI : 1.23 - 14.98], p = 0.022",
    #     ha="center",
    #     va="center",
    #     fontsize=22,
    # )

    # plt.savefig(
    #     f"{outcome}_km_curve",
    #     transparent=True,
    #     bbox_inches="tight",
    #     pad_inches=0,
    # )


def get_hazard_ratio(tesPD, predcol="finalPredB"):
    outcome = "Surv"
    label = "Label"

    # Fit a Cox Proportional Hazards model
    cph = CoxPHFitter()
    cph.fit(
        tesPD[[outcome, label, predcol]],
        duration_col=outcome,
        event_col=label,
    )
    cph.print_summary()

    # Calculate the hazard ratio between the two groups
    hazard_ratio = cph.hazard_ratios_[predcol]
    # print("Hazard Ratio:", hazard_ratio)

    return hazard_ratio


def combine_predictions(newdf, olddf):
    # _df = _df[_df.PatientID.str.contains("CLOG")]
    newdf = newdf.sort_values("PatientID")
    newdf["PatientID"] = newdf["PatientID"].astype(str)

    olddf = olddf.merge(newdf, on=["PatientID", "Label", "Surv", "Fold"])

    assert (newdf.PatientID.values == olddf.PatientID.values).all()

    pred1 = olddf["Pred_x"].values.tolist()
    pred2 = olddf["Pred_y"].values.tolist()

    pred = [pred1[i] + pred2[i] for i in range(len(pred1))]
    olddf["Pred"] = pred

    olddf = olddf[["PatientID", "Label", "Surv", "Fold", "Pred"]]

    olddf = olddf[newdf.columns]

    return olddf


def binarize_run_predictions(traindf, valdf, cvsplits=10):
    newvaldf = None

    for i in range(cvsplits):
        traindf_fold = traindf[traindf.Fold == i]
        valdf_fold = valdf[valdf.Fold == i]

        thresh = np.percentile(traindf_fold.Pred.values, (31 / 44) * 100)

        valdf_fold["PredB"] = valdf_fold.Pred.apply(
            lambda x: 0 if x <= thresh else 1
        )

        newvaldf = (
            valdf_fold
            if newvaldf is None
            else pd.concat([newvaldf, valdf_fold])
        )

    return newvaldf


if __name__ == "__main__":
    mainfolder = "../../../mlpipeline/outputs/RAPTOMICS/"

    ensemble_list = [
        # "rad/lassofeatsel_binpval_corr09_08-29-2023-07-58-22/feature_2/",
        # "path/lassofeatsel_nooutlier_stdonly_08-29-2023-07-34-24/feature_5/",
        # "path/NEW_cphfeatsel_nooutlier_stdonly_withtrainpred_10cv_300runs_08-30-2023-07-03-45/feature_2/",
        # "path/NEW_lassofeatsel_nooutlier_stdonly_withtrainpred_10cv_300runs_08-30-2023-07-08-06/feature_5/",
        "path/NEW_cphfeatsel_corr08_withtrainpred_10cv_300runs_08-30-2023-15-16-28/feature_4/",
        "rad/NEW_lassofeatsel_binpval_corr09_withtrainpred_10cv_300runs_08-30-2023-06-54-19/feature_3/",
    ]

    event_indicator = "Label"
    event_time = "Surv"

    finaldf = None

    for item in ensemble_list:
        files = (
            Path(mainfolder)
            .joinpath(ensemble_list[0])
            .joinpath("results")
            .glob("run*.csv")
        )

        cindices = []

        finalvaldf = None

        for fi in files:
            valdf = pd.read_csv(str(fi))
            valdf = valdf.sort_values("PatientID")
            valdf["PatientID"] = valdf["PatientID"].astype(str)

            traindf = pd.read_csv(str(fi).replace("run_", "TRAIN_run_"))
            traindf = traindf.sort_values("PatientID")
            traindf["PatientID"] = traindf["PatientID"].astype(str)

            for i in range(1, len(ensemble_list)):
                newpath = str(fi).replace(ensemble_list[0], ensemble_list[i])

                newvaldf = pd.read_csv(newpath)
                newtraindf = pd.read_csv(newpath.replace("run_", "TRAIN_run_"))

                valdf = combine_predictions(newvaldf, valdf)
                traindf = combine_predictions(newtraindf, traindf)

            valdf = binarize_run_predictions(traindf, valdf)
            valdf = valdf.sort_values("PatientID")

            if finalvaldf is None:
                finalvaldf = valdf[["PatientID", "Label", "Surv", "PredB"]]
            else:
                assert (
                    finalvaldf.PatientID.values == valdf.PatientID.values
                ).all()
                finalvaldf["PredB"] = finalvaldf["PredB"] + valdf["PredB"]

        hrs = []

        minpred = int(np.ceil(finalvaldf.PredB.min() / 10) * 10)
        maxpred = int(np.floor(finalvaldf.PredB.max() / 10) * 10)

        for finalthresh in range(minpred, maxpred, 5):
            # finalthresh = (13 / 44) * 300
            # # finalthresh = (22 / 44) * 300

            finalvaldf["finalPredB"] = finalvaldf.PredB.apply(
                lambda x: 0 if x <= finalthresh else 1
            )

            hrs.append((finalthresh, get_hazard_ratio(finalvaldf)))

        hrs = [x for x in hrs if x[1] < 100]

        max_tuple = max(hrs, key=lambda x: x[1])

        finalthresh = max_tuple[0]

        finalvaldf["finalPredB"] = finalvaldf.PredB.apply(
            lambda x: 0 if x <= finalthresh else 1
        )

        hr = get_hazard_ratio(finalvaldf)
        print(hr)

        plt_km_curve(
            finalvaldf[["Surv", "Label", "PredB"]],
            finalthresh,
            predcol="PredB",
        )

        # print(get_hazard_ratio(finalvaldf, predcol="PredB"))

        import pdb

        pdb.set_trace()
