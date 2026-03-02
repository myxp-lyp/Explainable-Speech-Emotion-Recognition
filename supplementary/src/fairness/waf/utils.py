from sklearn.preprocessing import OneHotEncoder

from fairness.constants import privileged_groups

def fit_one_hot_encoder(dataset, features):
    encoders = {}
    for key in features:
        values = dataset[key]
        enc = OneHotEncoder(sparse_output=False)
        enc.fit([[v] for v in values])
        encoders[key] = enc
    return encoders


# add feature combinations
def fit_combination_encoder(dataset, features):
    # Combine the features into (n_samples, len(features)) 2D array
    combo_values = list(zip(*(dataset[feat] for feat in features)))
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(combo_values)
    return encoder


def binarise_protected_attribute(batch, attribute):
    groups = privileged_groups.get(attribute, [])
    values = batch[attribute]
    encoded_vals = [1 if (value in groups) else -1 for value in values]
    batch[attribute] = encoded_vals
    return batch


def onehot_protected_attribute(batch, encoder, attribute):
    values = batch[attribute]
    encoded_vals = encoder.transform([[v] for v in values])
    batch[attribute] = encoded_vals
    return batch
