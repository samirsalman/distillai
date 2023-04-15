import torch
from distill_ai.modules.distillation_module import LogitsDistillationModule
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class KnowledgeDistillationTrainer(pl.Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(
        self,
        teacher_model: torch.nn.Module,
        student_model: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        student_target_loss: torch.nn.Module,
        distillation_loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        *args,
        **kwargs,
    ):
        distillation_module = LogitsDistillationModule(
            teacher_model=teacher_model,
            student_model=student_model,
            distillation_loss=distillation_loss,
            student_target_loss=student_target_loss,
            optimizer=optimizer,
            *args,
            **kwargs,
        )
        super().fit(
            distillation_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, *args, **kwargs
        )
