# load model and get the params
import pandas as pd

from databases.constants import id2label
from fairness.waf.config import WAFConfig
from fairness.waf.model import WAFModel
from utils import get_device


def parse_waf_metrics(waf_model: WAFModel, config: WAFConfig):
    device = get_device()

    outputs = []
    waf_model.eval()
    waf_model.to(device)

    columns = config.xds + config.xcs

    demographic_columns = [i for i in range(len(columns)) if columns[i] in config.xds]
    non_demographic_columns = [
        i for i in range(len(columns)) if columns[i] not in config.xds
    ]

    if config.use_one_hot or config.use_combination:
        print("--- Currently unsupported case, skipping for now ----")
        return pd.DataFrame()

    # Get the parameters
    params = list(waf_model.linear.weight.data.cpu().numpy())

    for emotion_id, coeffs in enumerate(params):
        assert len(coeffs) == len(columns)
        other_coeffs = [coeffs[i] for i in non_demographic_columns]
        for i in demographic_columns:
            outputs.append(
                {
                    "protected_attribute": columns[i],
                    "waf": coeffs[i],
                    "other_coeff_names": [columns[i] for i in non_demographic_columns],
                    "other_coeff_values": other_coeffs,
                    "emotion": id2label[emotion_id],
                }
            )

    outputs_df = pd.DataFrame(outputs)
    return outputs_df
