import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        temperature: float = 1.0,
        distillation_loss: nn.Module = nn.KLDivLoss(),
        *args,
        **kwargs,
    ):
        """Distillation loss

        Args:
            alpha (float, optional): The weight of the distillation loss. Defaults to 0.25.
            temperature (float, optional): The temperature of the distillation loss. Defaults to 1.0.
            distillation_loss (nn.Module, optional): The distillation loss function. Defaults to nn.KLDivLoss().
        """
        super().__init__(*args, **kwargs)
        self.distillation_loss = distillation_loss
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits: torch.Tensor, student_loss: torch.Tensor, teacher_logits: torch.Tensor):
        """Compute the distillation loss

        Args:
            student_logits (torch.Tensor): The logits of the student model
            student_target_loss (torch.Tensor): The target loss of the student model
            teacher_logits (torch.Tensor): The logits of the teacher model
        """
        distillation_loss = self.distillation_loss(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits.float() / self.temperature, dim=1),
        )

        loss = (1 - self.alpha) * student_loss + self.alpha * distillation_loss
        return loss
