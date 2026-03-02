import os
import argparse
from datasets import Dataset
from sklearn.metrics import classification_report


from databases.cremad import  extract_train_test_sets, load_db
from ser_models.evaluator import evaluate
from ser_models.hubert import HubertForSpeechClassification
from ser_models.speech_processing import preprocess_function
from ser_models.trainer import get_model, get_trainer
from ser_models.wavlm import WavLMForSpeechClassification
from utils import get_device, set_seed
import argparse


def train_all_models(cremad_root: str, model_output_dir: str, result_output_dir: str):
    set_seed()
    device = get_device()

    print(f"Using device: {device}")

    print(f"---- Loading CREMA-D dataset from: {cremad_root} ----")
    db = load_db(cremad_root)
    train_df, test_df = extract_train_test_sets(db)

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    models_info = {
        "hubert": {
            "name": "facebook/hubert-base-ls960",
            "class": HubertForSpeechClassification,
        },
        "wavlm": {
            "name": "microsoft/wavlm-base-plus",
            "class": WavLMForSpeechClassification,
        },
    }

    print(f"---- Processing audio data ----")
    train_dataset = train_dataset.map(
        preprocess_function, batch_size=100, batched=True, num_proc=4
    )
    test_dataset = test_dataset.map(
        preprocess_function, batch_size=100, batched=True, num_proc=4
    )
    

    for model_key, model_info in models_info.items():
        print(f"---- Training model: {model_key} ----")
        config, processor, model = get_model(
            model_info["name"], 
            model_info["class"]
        )
        model.to(device)
        model.freeze_feature_extractor()

        output_dir = os.path.join(model_output_dir, model_key)

        trainer = get_trainer(
            model,
            processor,
            output_dir=output_dir,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )

        trainer.train(resume_from_checkpoint=False)

        print(f"\n---- Evaluating model: {model_key} ----\n")

        model.eval()
        result = evaluate(model, processor, test_dataset)

        result.save_to_disk(os.path.join(result_output_dir, f"{model_key}_result"))

        label_names = [config.id2label[i] for i in range(config.num_labels)]
        print(f"Label names: {label_names}")

        y_true = [config.label2id[name] for name in result["emotion"]]
        y_pred = result["predicted"]

        print(y_true[:5])
        print(y_pred[:5])
        
def main(model_name: str, cremad_root: str, model_output_dir: str, result_output_dir: str):
    set_seed()
    device = get_device()

    print(f"Using device: {device}")

    print(f"---- Loading CREMA-D dataset from: {cremad_root} ----")
    db = load_db(cremad_root)
    train_df, test_df = extract_train_test_sets(db)

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # models_info = {
    #     "hubert": {
    #         "name": "facebook/hubert-base-ls960",
    #         "class": HubertForSpeechClassification,
    #     },
    #     "wavlm": {
    #         "name": "microsoft/wavlm-base-plus",
    #         "class": WavLMForSpeechClassification,
    #     },
    # }
    if model_name == "hubert":
        model_path = "facebook/hubert-base-ls960"
        model_class = HubertForSpeechClassification
    elif model_name == "wavlm":
        model_path = "microsoft/wavlm-base-plus"
        model_class = WavLMForSpeechClassification

    print(f"---- Processing audio data ----")
    train_dataset = train_dataset.map(
        preprocess_function, batch_size=100, batched=True, num_proc=4
    )
    test_dataset = test_dataset.map(
        preprocess_function, batch_size=100, batched=True, num_proc=4
    )

    print(f"---- Training model: {model_name} ----")
    config, processor, model = get_model(
        model_path, 
        model_class
    )
    model.to(device)
    model.freeze_feature_extractor()

    trainer = get_trainer(
        model,
        processor,
        output_dir=model_output_dir,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )

    trainer.train(resume_from_checkpoint=True)

    print(f"\n---- Evaluating model: {model_name} ----\n")

    model.eval()
    result = evaluate(model, processor, test_dataset)

    result.save_to_disk(result_output_dir)

    label_names = [config.id2label[i] for i in range(config.num_labels)]
    print(f"Label names: {label_names}")

    y_true = [config.label2id[name] for name in result["emotion"]]
    y_pred = result["predicted"]

    print(y_true[:5])
    print(y_pred[:5])
    print(classification_report(y_true, y_pred, target_names=label_names))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train SER models on CREMA-D dataset.")
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model Name e.g hubert, wavlm."
    )
    parser.add_argument(
        "--cremad_root", type=str, required=True, help="Path to the CREMA-D dataset root."
    )
    parser.add_argument(
        "--model_output_dir",
        type=str,
        default="./models",
        help="Directory to save trained models.",
    )
    parser.add_argument(
        "--result_output_dir",
        type=str,
        default="./result",
        help="Directory to save evaluation output.",
    )

    args = parser.parse_args()
    main(args.model_name, args.cremad_root, args.model_output_dir, args.result_output_dir)

    # train_all_models(args.cremad_root, args.model_output_dir, args.result_output_dir)