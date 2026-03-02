import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
from statsmodels.formula.api import ols


def calculate_mutual_information(df, target_variable, y_label="y"):

    xs = df[target_variable].unique()
    ys = df[y_label].unique()

    # marginal probabilities
    p_x = df[target_variable].value_counts(normalize=True)
    p_y = df[y_label].value_counts(normalize=True)

    # joint probabilities
    p_xy = df.groupby([y_label, target_variable]).size() / len(df)
    p_xy = p_xy.unstack(fill_value=0)

    # Calculate mutual information
    mutual_info = 0

    for x in xs:
        for y in ys:
            if p_xy[x][y] > 0:
                mutual_info += p_xy[x][y] * (
                    np.log(p_xy[x][y]) - np.log(p_x[x] * p_y[y])
                )
    return mutual_info


def calculate_mi_continuous(df, target_variables, y_label="y"):

    # Calculate mutual information using sklearn
    mi = mutual_info_regression(df[target_variables], df[y_label], random_state=1)
    return mi


def calculate_anova(df, attribute, emotion):
    model = ols(f"{emotion} ~ C({attribute})", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    sum_sq_class = anova_table["sum_sq"].iloc[0]  # Class variable sum of squares
    sum_sq_residual = anova_table["sum_sq"].iloc[1]  # Residual sum of squares
    r_2 = sum_sq_class / (sum_sq_class + sum_sq_residual)
    return r_2


def calculate_mi_df(dataset, mapping, attributes, emotions, y_label):
    dataset["is_correct"] = (dataset["predicted"] == dataset[y_label]).astype("float")

    outputs = []
    for j, emotion in enumerate(emotions):
        mi = calculate_mi_continuous(dataset, attributes, emotion)
        mi_bin = calculate_mi_continuous(
            dataset[dataset["emotion"] == emotion], attributes, "is_correct"
        )
        for i, mi_value in enumerate(mi):
            anova_result = calculate_anova(dataset, attributes[i], emotion)
            contributing_attributes = mapping[j]
            raw_contibution = (
                1 / len(contributing_attributes)
                if attributes[i] in contributing_attributes
                else 0
            )
            outputs.append(
                {
                    "protected_attribute": attributes[i],
                    y_label: emotion,
                    "mi": mi[i],
                    "mi_bin": mi_bin[i],
                    "r_squared": anova_result,
                    "raw_contribution": raw_contibution,
                }
            )
    mi_df = pd.DataFrame(outputs)
    return mi_df


def calculate_correlations(df):

    outputs = []
    for fairness_proxy in ["mi", "mi_bin", "r_squared", "raw_contribution"]:
        print(f"---- Correlation Results With {fairness_proxy}----")
        for metric in [
            "Statistical Parity",
            "Equal Opportunity",
            "False Positive Rate",
            "waf",
        ]:
            for method in ["pearson", "spearman"]:
                if method == "pearson":
                    corr, p_value = pearsonr(df[metric], df[fairness_proxy])
                else:
                    corr, p_value = spearmanr(df[metric], df[fairness_proxy])

                print(
                    f"{method.capitalize()} correlation between {metric} and {fairness_proxy}: {corr:.4f}, p-value: {p_value:.4f}"
                )
                outputs.append(
                    {
                        "metric": metric,
                        "fairness_proxy": fairness_proxy,
                        "method": method,
                        "corr": corr,
                        "p_value": p_value,
                    }
                )
    outputs_df = pd.DataFrame(outputs)

    return outputs_df
