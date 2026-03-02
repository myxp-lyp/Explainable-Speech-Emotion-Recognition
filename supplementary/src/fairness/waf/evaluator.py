import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader

from fairness.waf.dataset import collator_fn
from utils import get_device


def evaluate_waf(model, dataset):
    """
    Evaluate the WAF model on the provided dataset.

    """
    device = get_device()
    model.eval()
    model = model.to(device)
    y_true = []
    y_pred = []

    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False, collate_fn=collator_fn
    )

    for batch in dataloader:
        x = batch["waf_features"].float().to(device)
        y = batch["waf_label"].float().to(device)

        with torch.no_grad():
            y_hat = model(x)

        y_true.extend(y.cpu().numpy())
        y_pred.extend(y_hat.cpu().numpy())

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mse, r2
