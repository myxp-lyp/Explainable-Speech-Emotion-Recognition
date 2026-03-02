import torch
import torchaudio
import librosa
import numpy as np
import torch.nn.functional as F

def speech_file_to_array_fn(batch, target_sr=16000):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = speech_array.squeeze().numpy()
    speech_array = librosa.resample(np.asarray(speech_array), orig_sr= sampling_rate, target_sr=target_sr)

    batch["speech"] = speech_array
    return batch


def predict(batch, model, feature_extractor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features = feature_extractor(
        batch["speech"],
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
    )

    input_values = features.input_values.to(device)
    attention_mask = (
        features.attention_mask.to(device) if "attention_mask" in features else None
    )

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    scores = F.softmax(logits, dim=1).detach().cpu().numpy()
    batch["predicted"] = pred_ids
    batch["scores"] = scores
    return batch


def evaluate(model, processor, test_dataset):

    test_dataset = test_dataset.map(speech_file_to_array_fn)

    result = test_dataset.map(
        predict,
        batched=True,
        batch_size=1,
        fn_kwargs={"model": model, "feature_extractor": processor},
    )
    return result
