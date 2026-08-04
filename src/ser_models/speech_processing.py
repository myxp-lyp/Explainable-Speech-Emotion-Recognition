import torchaudio
from databases.constants import labels as _DEFAULT_LABELS


def speech_file_to_array_fn(path, target_sr):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(
        sampling_rate, target_sr
    )  
    speech = resampler(speech_array).squeeze().numpy()
    return speech


def label_to_id(label, label_list):
    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label


def preprocess_function(
    examples, target_sr=16000, input_column="path", output_column="emotion",
    label_list=None,
):
    """Load audio and convert emotion strings to indices.

    ``label_list`` overrides the default CREMA-D label ordering; needed for
    IEMOCAP (4-class) or any other dataset.
    """
    labels = label_list if label_list is not None else _DEFAULT_LABELS
    speech_list = [speech_file_to_array_fn(path, target_sr) for path in examples[input_column]]
    target_list = [
        label_to_id(label, labels) for label in examples[output_column]
    ]

    return {
        "input_values": speech_list,  # raw audio; to be processed in collator
        "labels": target_list,
    }
