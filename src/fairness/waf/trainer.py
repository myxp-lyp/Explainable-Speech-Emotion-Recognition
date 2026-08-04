import os

import torch
from sklearn.metrics import r2_score
from torch import nn, optim
from torch.utils.data import DataLoader


class WAFTrainer:
    def __init__(self, model, config, dataset, collator_fn):
        self.config = config
        self.dataset = dataset
        self.collator_fn = collator_fn
        self.device = torch.device(config.device)
        self.cache_path = os.path.join(config.model_cache_dir, f"waf_model.pth")

        os.makedirs(config.model_cache_dir, exist_ok=True)
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()
        self.dataloader = DataLoader(
            dataset, batch_size=config.batch_size, collate_fn=collator_fn
        )

    def train(self):
        if os.path.exists(self.cache_path) and self.config.use_model_cache:
            print(f"loading cached model: {self.cache_path}")
            self.model.load_state_dict(torch.load(self.cache_path))
            return

        best_mse, best_r2 = float("inf"), -float("inf")
        patience = 5
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            self.model.train()
            y_trues, y_preds, total_loss = [], [], 0

            for batch in self.dataloader:
                self.optimizer.zero_grad()
                x = batch["waf_features"].float().to(self.device)
                y = batch["waf_label"].float().to(self.device)

                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)

                loss.backward()
                self.optimizer.step()

                y_preds.extend(y_hat.detach().cpu().numpy())
                y_trues.extend(y.detach().cpu().numpy())
                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            r2 = r2_score(y_trues, y_preds)

            if avg_loss < best_mse:
                best_mse = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping.")
                    break

            best_r2 = max(best_r2, r2)

            print(
                f"Epoch {epoch+1}/{self.config.num_epochs} - Loss: {avg_loss:.4f}, R2: {r2:.4f}"
            )
        print(
            f"Best MSE: {best_mse:.4f}, Best R2: {best_r2:.4f} after {self.config.num_epochs} epochs"
        )
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        print(f"Saving model to {self.cache_path}")
        torch.save(self.model.state_dict(), self.cache_path)
