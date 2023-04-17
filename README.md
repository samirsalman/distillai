# DistillAI

## About the project

<p align="center">

<img src="https://user-images.githubusercontent.com/33979978/232232093-8bc9ab84-4c10-44da-becc-bbe21571c63c.png" width="240px">

</p>
DistillAI is a PyTorch library for knowledge distillation. It is built on top of PyTorch Lightning and provides a simple API for training knowledge distillation models.

## Installation

1. Clone the repo
2. Intall using pip `pip install -e .`

## Usage

Look at https://github.com/samirsalman/distillai/blob/main/examples/example.py for more details.

```python
from distill_ai.losses.distillation_loss import DistillationLoss
from distill_ai.trainers.distillation_trainer import KnowledgeDistillationTrainer

# init student and teacher models
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
    # pytorch lightning kwargs
)
# train
trainer.fit(
    # torch data loaders
    train_dataloader=train_dataloader,
    val_dataloader=test_dataloader,
    # models
    teacher_model=teacher,
    student_model=student,
    # loss functions
    student_target_loss=student_target_loss,
    distillation_loss=distillation_loss,
    # optimizer
    optimizer=optimizer,
)

```

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 
