# compute DP, EO, EOD
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

from databases.constants import demographic_columns, id2label, label2id
from fairness.constants import privileged_groups


def calculate_traditional_fairness_metrics(df):

    outputs = []
    for col in demographic_columns:
        groups = privileged_groups[col]
        encoded_vals = df[col].apply(lambda x: 1 if x in groups else -1).astype(int)
        df[col] = encoded_vals

    for id in label2id.values():
        # print(f"\n===== Fairness Metrics for {model_name} =====")
        # Convert the dataset to a BinaryLabelDataset
        df["y_true"] = (df["emotion"].map(label2id) == id).astype(int)
        df["y_pred"] = (df["predicted"] == id).astype(int)

        if df["y_true"].sum() == 0:
            print(
                f"Skipping class {id} ({id2label[id]}) due to no positives in ground truth"
            )
            continue

        for col in demographic_columns:

            for group_val in [1, -1]:
                subset = df[df[col] == group_val]
                if subset["y_true"].sum() == 0:
                    print(
                        f"Group {group_val} has no positives for {col} - may cause EOD to be undefined."
                    )

            binary_dataset = BinaryLabelDataset(
                df=df[["y_true", col]],
                label_names=["y_true"],
                protected_attribute_names=[col],
            )

            classified_dataset = binary_dataset.copy()
            classified_dataset.labels = df["y_pred"]

            binary_privileged_groups = [{col: [1]}]
            binary_unprivileged_groups = [{col: [-1]}]

            metric = ClassificationMetric(
                binary_dataset,
                classified_dataset,
                privileged_groups=binary_privileged_groups,
                unprivileged_groups=binary_unprivileged_groups,
            )

            statistical_parity = metric.statistical_parity_difference()
            equal_opportunity = metric.equal_opportunity_difference()
            false_positive_rate = metric.false_positive_rate_difference()

            metric_output = {
                "Statistical Parity": statistical_parity,
                "Equal Opportunity": equal_opportunity,
                "False Positive Rate": false_positive_rate,
            }

            metric_output["emotion"] = id2label[id]
            metric_output["protected_attribute"] = col
            outputs.append(metric_output)

    outputs_df = pd.DataFrame(outputs)
    return outputs_df
