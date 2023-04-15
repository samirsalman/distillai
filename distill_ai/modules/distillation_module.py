import torch
from typing import Any, List, Tuple
import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics
from distill_ai.losses.distillation_loss import DistillationLoss


class BaseKnowledgeDistillationModule(pl.LightningModule):
    def __init__(
        self,
        teacher_model: torch.nn.Module,
        student_model: torch.nn.Module,
        distillation_loss: torch.nn.Module,
        student_target_loss: torch.nn.Module = nn.CrossEntropyLoss(),
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.student = student_model
        self.distillation_loss = distillation_loss
        self.student_target_loss = student_target_loss
        self.optimizer = optimizer
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def training_step(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def validation_step(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError


class LogitsDistillationModule(BaseKnowledgeDistillationModule):
    def __init__(
        self,
        teacher_model: torch.nn.Module,
        student_model: torch.nn.Module,
        distillation_loss: torch.nn.Module = DistillationLoss(),
        student_target_loss: torch.nn.Module = nn.CrossEntropyLoss(),
        lr: float = 1e-3,
        temperature: float = 1.0,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        metrics: List[torchmetrics.Metric] = [torchmetrics.Accuracy()],
        *args: Any,
        **kwargs: Any,
    ):
        self.lr = lr
        self.temperature = temperature
        self.metrics = metrics
        super().__init__(
            teacher_model, student_model, distillation_loss, student_target_loss, optimizer, *args, **kwargs
        )

    def forward(self, x):
        # Forward pass of the student model
        return self.student(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Unpack the batch
        x, y = batch
        # Compute the logits of the teacher model
        with torch.no_grad():
            teacher_logits = self.teacher(x) / self.temperature
        # Compute the logits of the student model
        student_logits = self.student(x) / self.temperature
        # Compute the classification loss
        classification_loss = self.classification_loss_fn(student_logits, y)

        # Compute the distillation loss
        distillation_loss = self.distillation_loss_fn(
            student_logits=student_logits, student_loss=classification_loss, teacher_logits=teacher_logits
        )
        # Compute the total loss
        loss = distillation_loss + classification_loss
        for metric in self.metrics:
            # Log the metrics
            result = metric(student_logits, y.int())
            self.log(f"train_{metric._get_name().lower()}", result, prog_bar=True)
        # Log the losses
        self.log("train_distillation_loss", distillation_loss)
        self.log("train_classification_loss", classification_loss)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack the batch
        x, y = batch
        # Compute the logits of the student model
        logits = self.student(x)
        # Compute the classification loss
        loss = self.classification_loss_fn(logits, y)
        # Log the loss
        self.log("val_loss", loss, prog_bar=True)

        for metric in self.metrics:
            # Log the metrics
            result = metric(logits, y.int())
            self.log(f"val_{metric._get_name().lower()}", result, prog_bar=True)

        return loss

    def classification_loss_fn(self, logits, targets):
        # Define the classification loss function here
        return self.student_target_loss(logits.float(), targets.long())

    def distillation_loss_fn(self, student_logits, student_loss, teacher_logits):
        # Define the distillation loss function here
        return self.distillation_loss(student_logits, student_loss, teacher_logits)

    def configure_optimizers(self):
        # Define the optimizer and learning rate scheduler here
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer
