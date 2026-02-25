# trainer.py
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from dataclasses import dataclass
from accelerate import Accelerator
from src.utils.metrics import MetricsLogger

@dataclass
class TrainStats:
    train_loss: float
    train_acc: float
    clean_acc: float
    asr: float

class Trainer:
    def __init__(
        self, 
        model, 
        optimizer, 
        scheduler, 
        train_loader, 
        clean_loader, 
        poison_loader, 
        accelerator: Accelerator, 
        cfg
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.clean_loader = clean_loader
        self.poison_loader = poison_loader
        self.accelerator = accelerator
        self.cfg = cfg
        
        self.criterion = nn.CrossEntropyLoss()
        self.target_label = cfg.attack.target_label

        self.global_step = 0
        self.save_dir = os.path.join(cfg.checkpoints_dir, cfg.dataset.name)
        
        # 实例化日志管理器
        self.logger = MetricsLogger(self.save_dir, self.accelerator.is_main_process)
        
        self.best_clean_acc = 0.0
        self.best_asr = 0.0
        self.start_epoch = 0

    def load_checkpoint(self):
        """尝试从 latest_model.pth 恢复断点"""
        ckpt_path = os.path.join(self.save_dir, "latest_model.pth")
        
        if os.path.exists(ckpt_path):
            self.logger.info(f"🔄 Loading checkpoint from {ckpt_path} ...")
            # 必须使用 map_location="cpu" 防止多卡 OOM
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            
            # 剥离外壳加载模型
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_state_dict(checkpoint["model"])
            
            # 加载其他状态
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.best_clean_acc = checkpoint["best_clean_acc"]
            self.best_asr = checkpoint["best_asr"]
            self.start_epoch = checkpoint["epoch_current"] + 1
            
            self.logger.success(f"✅ Successfully resumed from Epoch {self.start_epoch}")
        else:
            self.logger.info("⚠️ No checkpoint found. Starting from scratch.")

    def save_checkpoint(self, epoch: int, is_best: bool):
        """统一的保存逻辑，使用 accelerator.save 防止多进程冲突"""
        # 同步所有进程，防止在保存前某一张卡跑得太快
        self.accelerator.wait_for_everyone()
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        state_dict = {
            "model": unwrapped_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_clean_acc": self.best_clean_acc,
            "best_asr": self.best_asr, 
            "epoch_current": epoch,
        }
        
        # 1. 永远保存一个 latest 供断点续训
        latest_path = os.path.join(self.save_dir, "latest_model.pth")
        self.accelerator.save(state_dict, latest_path)
        
        # 2. 如果是最佳指标，额外存一个 best
        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pth")
            self.accelerator.save(state_dict, best_path)
            self.logger.save_results_json(self.best_clean_acc, self.best_asr, epoch + 1)
            self.logger.success(f"🌟 New best checkpoint saved to {best_path}")

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Training", disable=not self.accelerator.is_main_process, mininterval=2.0)
        for batch in pbar:
            self.optimizer.zero_grad()
            inputs = batch["pixel_values"]
            targets = batch["label"]
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            self.accelerator.backward(loss)
            self.optimizer.step()

            self.logger.log_train_step(loss.item(), self.global_step)
            self.global_step += 1
            
            total_loss += loss.item() * inputs.size(0)
            predictions = torch.argmax(outputs, dim=1)
            total_correct += (predictions == targets).sum().item()
            total_samples += inputs.size(0)
            
            avg_loss = total_loss / total_samples
            avg_acc = (total_correct / total_samples) * 100.0
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "Acc": f"{avg_acc:.2f}%"})
            
        self.scheduler.step()
        return avg_loss, avg_acc

    def evaluate_clean(self):
        self.model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch in tqdm(self.clean_loader, desc="Eval Clean", leave=False, disable=not self.accelerator.is_main_process):
                inputs = batch["pixel_values"]
                targets = batch["label"]
                outputs = self.model(inputs)
                predictions = torch.argmax(outputs, dim=1)

                predictions, targets = self.accelerator.gather_for_metrics((predictions, targets))
                total_correct += (predictions == targets).sum().item()
                total_samples += inputs.size(0)
        return (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0

    def evaluate_asr(self):
        self.model.eval()
        attack_success = 0
        valid_samples = 0
        with torch.no_grad():
            for batch in tqdm(self.poison_loader, desc="Eval ASR", leave=False, disable=not self.accelerator.is_main_process):
                inputs = batch["pixel_values"]
                targets = batch["label"]           
                orig_labels = batch["orig_labels"] 
                outputs = self.model(inputs)
                predictions = torch.argmax(outputs, dim=1)

                predictions, targets, orig_labels = self.accelerator.gather_for_metrics(
                    (predictions, targets, batch["orig_labels"]))
                
                mask = (orig_labels != self.target_label)
                if mask.sum() > 0:
                    valid_predictions = predictions[mask]
                    valid_targets = targets[mask]
                    attack_success += (valid_predictions == valid_targets).sum().item()
                    valid_samples += mask.sum().item()
        return (attack_success / valid_samples) * 100.0 if valid_samples > 0 else 0.0

    def run(self):
        self.load_checkpoint()
        
        self.logger.info(f"🚀 Starting training from epoch {self.start_epoch} to {self.cfg.epochs}...")
        
        for epoch in range(self.start_epoch, self.cfg.epochs):
            train_loss, train_acc = self.train_one_epoch(epoch)
            clean_acc = self.evaluate_clean()
            asr = self.evaluate_asr()
            
            # 使用外部 logger 记录 Epoch 级别信息
            self.logger.log_epoch_metrics(epoch + 1, self.cfg.epochs, train_loss, train_acc, clean_acc, asr)
            
            # 判断是否是最优模型
            is_best = False
            if clean_acc > self.best_clean_acc or (clean_acc > self.best_clean_acc - 0.1 and asr > self.best_asr):
                self.best_clean_acc = clean_acc
                self.best_asr = asr
                is_best = True
            
            # 调用封装好的 save_checkpoint (每个 epoch 都会存 latest，遇到 best 会额外存 best)
            self.save_checkpoint(epoch, is_best)
            
        self.logger.close()