import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from distill_ai.losses import DistillationLoss
from distill_ai.trainers import KnowledgeDistillationTrainer

try:
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
except ImportError:
    raise ImportError("Please install sklearn to run the example: pip install sklearn")

# number of classes
N_CLASSES = 4


# define student and teacher models
class Student(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, N_CLASSES),
        )

    def forward(self, x):
        return self.model(x)


class Teacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, x):
        return self.model(x)


# create dataloaders
## create dataset
dataset = make_classification(n_samples=20000, n_features=784, n_classes=N_CLASSES, n_informative=32, n_redundant=0)
## split into train and test
train_x, test_x, train_y, test_y = train_test_split(torch.Tensor(dataset[0]), torch.Tensor(dataset[1]), test_size=0.2)
train_dataset = list(zip(train_x, train_y))
test_dataset = list(zip(test_x, test_y))
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)


student = Student()
teacher = Teacher()
# define loss functions
student_target_loss = nn.CrossEntropyLoss()
# define distillation loss
distillation_loss = DistillationLoss(alpha=0.25, temperature=1.0)
# define optimizer
optimizer = torch.optim.Adam
# define trainer
trainer = KnowledgeDistillationTrainer(
    max_epochs=10,
)
# train
trainer.fit(
    train_dataloader=train_dataloader,
    val_dataloader=test_dataloader,
    teacher_model=teacher,
    student_model=student,
    student_target_loss=student_target_loss,
    distillation_loss=distillation_loss,
    optimizer=optimizer,
)
