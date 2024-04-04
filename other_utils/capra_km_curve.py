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


def get_hazard_ratio(tesPD, predcol="PredB"):
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

    return newvaldf


if __name__ == "__main__":
    labelsdf = pd.read_csv(
        "../outputs/csvs/labels/finalccfbcrlabelsJune2021.csv"
    )

    traindf = pd.read_csv("../outputs/csvs/features/DE_radiomic_ALL.csv")
    traindf = traindf[["PatientID", "CAPRA"]]

    traindf = traindf.drop_duplicates()

    labelsdf = labelsdf[labelsdf.Phase == "train"]

    labelsdf = labelsdf.merge(traindf, on="PatientID")

    scores = labelsdf.CAPRA.values
    # thresh = np.percentile(scores, (31 / 44) * 100)
    thresh = 3

    labelsdf = labelsdf[
        ["PatientID", "RisingPSALabel", "timeDiffMonths", "CAPRA"]
    ]

    labelsdf["PredB"] = labelsdf["CAPRA"].apply(
        lambda x: 1 if x > thresh else (0 if x <= thresh else x)
    )

    labelsdf = labelsdf.set_index("PatientID")
    labelsdf = labelsdf.rename(
        columns={
            "RisingPSALabel": "Label",
            "timeDiffMonths": "Surv",
        }
    )

    # labelsdf = labelsdf.drop("CAPRA", axis=1)

    hr = get_hazard_ratio(labelsdf)
    print(hr)

    plt_km_curve(
        labelsdf[["Surv", "Label", "CAPRA"]],
        thresh,
        predcol="CAPRA",
    )

    # print(get_hazard_ratio(finalvaldf, predcol="PredB"))

    import pdb

    pdb.set_trace()
