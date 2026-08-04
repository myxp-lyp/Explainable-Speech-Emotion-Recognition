import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from databases.constants import demographic_columns, labels
from databases.synthetic.constants import THRESHOLD_K, mappings
from databases.synthetic.mutual_info import (calculate_anova,
                                             calculate_mi_continuous)
from utils import set_seed

set_seed()

low_prob_range = (0.1, 0.3)
high_prob_range = (0.7, 0.9)
med_prob_range = (0.3, 0.5)

priv = 1  # group to bias against (privileged group)


def _bias_triggers(row, attributes, regime: str, k: int):
    """Return True iff bias should be injected for a given emotion whose
    associated attribute subset is ``attributes``.

    regime == 'additive' / 'multi' / 'pairwise' : trigger iff every attr in
                                                   the subset is privileged
                                                   (=1). Original SEAB rule.
    regime == 'threshold'                       : trigger iff at least ``k``
                                                   of the 4 demographic attrs
                                                   are privileged, AND the
                                                   emotion has a non-empty
                                                   subset.
    regime == 'xor'                             : trigger iff *exactly one*
                                                   of the pair is privileged.
                                                   Requires len(attributes)==2.
                                                   Zero main-effect signal --
                                                   only pairwise interaction
                                                   terms can detect this.
    """
    if len(attributes) == 0:
        return False
    if regime == "threshold":
        n_priv = sum(1 for a in demographic_columns if row[a] == priv)
        return n_priv >= k
    if regime == "xor":
        assert len(attributes) == 2, "XOR regime requires exactly 2 attributes per emotion"
        return (row[attributes[0]] == priv) != (row[attributes[1]] == priv)
    # additive / multi / pairwise (SEAB original semantics: AND over subset)
    return all(row[a] == priv for a in attributes)


def generate_pred_scores(row, mapping, regime: str = "multi", k: int = THRESHOLD_K):
    y = np.zeros(6)  # Initialize y with zeros to minimise error as much as possible
    true = row["true"]
    true_index = np.argmax(true)

    y[true_index] = np.random.uniform(
        *med_prob_range
    )  # Set highest probability so far for true class

    for emotion_index, attributes in mapping.items():
        if len(attributes) == 0 and true[emotion_index] == 1:
            # if emotion has no attributes then set high probability
            y[emotion_index] = np.random.uniform(*high_prob_range)
            continue

        if _bias_triggers(row, attributes, regime, k):
            # if emotion is true class then set low probability
            if true[emotion_index] == 1:
                y[emotion_index] = np.random.uniform(*low_prob_range)
                # for other classes set to higher probability
                for i in range(6):
                    if (
                        i != emotion_index and y[i] < y[emotion_index]
                    ):  # ensure this class isnt lowest
                        y[i] = np.random.uniform(*med_prob_range)
            else:
                # set high probability for false class
                y[emotion_index] = np.random.uniform(*high_prob_range)
    T = 0.1
    y = y ** (1 / T) / (np.sum(y ** (1 / T)))  # L1 normalization
    return y


def generate_true(row):
    true = np.zeros(6)
    true_index = np.random.randint(0, 6)
    true[true_index] = 1

    return true


def compute_loss(row):
    true = row["true"]
    pred = row["pred_scores"]

    eps = 1e-10 

    # Calculate BCE per class
    loss = -(np.log(pred + eps) * true + (1 - true) * np.log(1 - pred + eps))
    return loss


# add gt contributions of attributes to the dataset
def calculate_contributions(dataset, mapping):
    dataset["is_correct"] = (dataset["predicted"] == dataset["emotion"]).astype("float")
    losses = dataset["loss"].apply(lambda x: x.tolist())
    losses = pd.DataFrame(losses.tolist(), columns=labels)

    dataset = pd.concat([dataset, losses], axis=1)

    all_contributions = []

    for emotion_index, attr in mapping.items():
        emotion = labels[emotion_index]
        mis = calculate_mi_continuous(dataset, demographic_columns, emotion)
        mi_bin = calculate_mi_continuous(
            dataset[dataset["emotion"] == emotion], demographic_columns, "is_correct"
        )

        for i, attribute in enumerate(demographic_columns):
            contribution = 1 / len(attr) if attribute in attr else 0
            anova_result = calculate_anova(dataset, attribute, emotion)
            all_contributions.append(
                {
                    "emotion": labels[emotion_index],
                    "attribute": attribute,
                    "contribution": contribution,
                    "mi": mis[i],
                    "anova": anova_result,
                }
            )

    contributions_df = pd.DataFrame(all_contributions)
    return contributions_df


def calculate_correlations(df):
    outputs = []
    for fairness_proxy in ["mi", "anova"]:
        # print(f'---- Correlation Results With {fairness_proxy}----')
        for method in ["pearson", "spearman"]:
            if method == "pearson":
                corr, p_value = pearsonr(df["contribution"], df[fairness_proxy])
            else:
                corr, p_value = spearmanr(df["contribution"], df[fairness_proxy])

            outputs.append(corr)
    return outputs


def generate_dataset(mapping, output_csv, regime: str = "multi", k: int = THRESHOLD_K):
    truth_table = list(itertools.product([-1, 1], repeat=len(demographic_columns)))
    full_table = truth_table * 100

    dataset = pd.DataFrame(full_table, columns=demographic_columns)

    dataset["true"] = dataset.apply(generate_true, axis=1)
    dataset["pred_scores"] = dataset.apply(
        lambda row: generate_pred_scores(row, mapping=mapping, regime=regime, k=k),
        axis=1,
    )
    dataset["loss"] = dataset.apply(compute_loss, axis=1)

    dataset["emotion"] = dataset["true"].apply(lambda x: labels[np.argmax(x)])
    dataset["predicted"] = dataset["pred_scores"].apply(lambda x: labels[np.argmax(x)])

    if output_csv:
        dataset.to_csv(output_csv, index=False)

    return dataset


def objective_function(dataset):

    best_low_prob = 0
    best_high_prob = 0
    best_correlation_sum = -np.inf
    best_correlation = []

    results = []

    low_probs = np.linspace(0, 1, 10)  # Create 10 evenly spaced edges between 0 and 1
    high_probs = np.linspace(0, 1, 10)  # Create 10 evenly spaced edges between 0 and 1

    for low_prob in low_probs:
        for high_prob in high_probs:
            if low_prob >= high_prob:
                continue

            # Update the dataset with new error ranges
            dataset = generate_dataset(dataset)
            dataset = dataset.drop(columns=labels)
            dataset = pd.concat(
                [dataset, pd.DataFrame(dataset["loss"].tolist(), columns=labels)],
                axis=1,
            )
            contributions_df = calculate_contributions(dataset)
            correlations = calculate_correlations(contributions_df)
            print(
                f"low probability threshold: {low_prob}, high probability threshold: {high_prob}"
            )
            print(f"Correlations: {correlations}")

            corr_sum = sum(correlations)

            result = {
                "low_prob": low_prob,
                "high_prob": high_prob,
                "mi_pearson": correlations[0],
                "mi_spearman": correlations[1],
                "anova_pearson": correlations[2],
                "anova_spearman": correlations[3],
                "correlation_sum": corr_sum,
            }
            results.append(result)

            if corr_sum > best_correlation_sum:
                best_correlation_sum = corr_sum
                best_correlation = correlations
                best_low_prob = low_prob
                best_high_prob = high_prob

                print(
                    f"Best correlation sum so far: {best_correlation_sum} with correlations: {correlations}"
                )

    print(
        f"Best low prob: {best_low_prob}, best high prob: {best_high_prob}, Best correlation sum: {best_correlation_sum}, Best correlations: {best_correlation}"
    )
    results_df = pd.DataFrame(results)
    return results_df


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a biased dataset")
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save the generated dataset (CSV)",
    )
    parser.add_argument(
        "--mapping",
        type=str,
        required=True,
        help="Attribute-emotion mapping: 'single'/'additive', 'multi', 'threshold'",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default=None,
        choices=["additive", "multi", "threshold"],
        help="Bias trigger regime; defaults derived from --mapping",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=THRESHOLD_K,
        help="Threshold k for the 'threshold' regime (default: 3)",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    mapping = mappings[args.mapping]
    output_csv = args.output_csv

    if args.regime is None:
        regime = "threshold" if args.mapping == "threshold" else \
                 ("additive" if args.mapping in ("single", "additive") else "multi")
    else:
        regime = args.regime

    print(f"Generating dataset with mapping={args.mapping} regime={regime} k={args.k}")
    dataset = generate_dataset(mapping, output_csv, regime=regime, k=args.k)
    print(f"Dataset saved to {output_csv}")

    print("Calculating mutual information...")

    contributions_df = calculate_contributions(dataset, mapping)
    print(contributions_df)

    print(dataset.head())
