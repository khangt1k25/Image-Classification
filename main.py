from matplotlib.pyplot import bar
from Model.trainer import Trainer 

trainer = Trainer(
    model_name='lenet',
    batch_size=64,
    lr=1e-3,
    n_epochs=100
)

trainer.fit()
trainer.save_model(trainer.n_epochs)
trainer.plot_history()