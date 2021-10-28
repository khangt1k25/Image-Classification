from matplotlib.pyplot import bar
from Model.trainer import Trainer 

trainer = Trainer(
    model_name='alexnet',
    lr=5e-4,
    n_epochs=100,
    batch_size=32,
    use_pretrained=True,
    training_data_percent=0.9
)

trainer.fit()
trainer.save_model(trainer.n_epochs)
trainer.plot_history()