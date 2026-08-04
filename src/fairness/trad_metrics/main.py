import pandas as pd
import datasets
from fairness.trad_metrics.trad_metrics import calculate_traditional_fairness_metrics
import argparse
import os
from IPython.display import display

def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate traditional fairness metrics")
    parser.add_argument(
        "--ser_result_dataset",
        type=str,
        required=True,
        help="Path to SER model dataset",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,        
        help="Path to save the fairness metrics CSV file",
    )
    return parser.parse_args()

def main(ser_result_dataset, output_csv):

    metrics_df = pd.DataFrame()
    dataset = datasets.load_from_disk(ser_result_dataset)

    df = dataset.to_pandas()

    outputs_df = calculate_traditional_fairness_metrics(df)

    metrics_df = pd.concat([metrics_df, outputs_df], axis=0)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    metrics_df.to_csv(output_csv, index=False)
    display(metrics_df)


if __name__ == "__main__":
    args = parse_arguments()
    ser_result_dataset = args.ser_result_dataset
    output_csv = args.output_csv
    main(ser_result_dataset, output_csv)