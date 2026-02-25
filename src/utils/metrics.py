import os
import sys
import json
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

class MetricsLogger:
    def __init__(self, save_dir: str, is_main_process: bool):
        self.save_dir = save_dir
        self.is_main_process = is_main_process
        self.tb_writer = None
        self.history = []

        if self.is_main_process:
            os.makedirs(self.save_dir, exist_ok=True)
            self._setup_logger()
            self.tb_writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "tb_logs"))

    def _setup_logger(self):
        logger.remove()
        logger.add(
            sys.stderr, 
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
        )
        logger.add(os.path.join(self.save_dir, "training.log"), rotation="10 MB")

    def info(self, msg: str):
        if self.is_main_process:
            logger.info(msg)

    def success(self, msg: str):
        if self.is_main_process:
            logger.success(msg)

    def warning(self, msg: str):
        if self.is_main_process:
            logger.warning(msg)

    def log_train_step(self, loss: float, step: int):
        if self.is_main_process and self.tb_writer:
            self.tb_writer.add_scalar("Train/Step_Loss", loss, step)

    def log_epoch_metrics(self, epoch: int, total_epochs: int, train_loss: float, train_acc: float, clean_acc: float, asr: float):
        if self.is_main_process:
            self.info(
                f"Epoch [{epoch}/{total_epochs}] | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Clean Acc: {clean_acc:.2f}% | ASR: {asr:.2f}%"
            )
            if self.tb_writer:
                self.tb_writer.add_scalar("Train/Loss", train_loss, epoch)
                self.tb_writer.add_scalar("Train/Accuracy", train_acc, epoch)
                self.tb_writer.add_scalar("Eval/Clean_Accuracy", clean_acc, epoch)
                self.tb_writer.add_scalar("Eval/ASR", asr, epoch)
            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "clean_acc": clean_acc,
                "asr": asr
            })

    def save_results_json(self, best_clean_acc: float, best_asr: float, epoch: int):
        if self.is_main_process:
            results = {"best_clean_acc": best_clean_acc, "best_asr": best_asr, "best_epoch": epoch}
            with open(os.path.join(self.save_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=4)

            try:
                import pandas as pd
                pd.DataFrame(self.history).to_csv(os.path.join(self.save_dir, "metrics_raw.csv"), index=False)
                self.success("Raw metrics saved to metrics_raw.csv")
            except ImportError:
                pass

    def close(self):
        if self.is_main_process and self.tb_writer:
            self.tb_writer.close()
            self.info("🎉 Training completed and TensorBoard writer closed!")