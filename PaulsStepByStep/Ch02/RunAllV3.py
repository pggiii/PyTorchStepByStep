import numpy as np
import DataGenerationV1 as dg
import DataPreparationV2 as dp
import ModelConfigurationV4 as mc
import ModelTrainingV5 as mt
import StepByStep

x, y = dg.RunDataGeneration(showPlot=False)

train_loader, val_loader =  dp.RunDataPreparation(x, y)

model, optimizer, loss_fn = mc.RunModelConfiguration()

sbs = mt.RunModelTraining(model, loss_fn, optimizer, train_loader, val_loader, plot_losses=True)

# Make Predictions
#

new_data = np.array([.5, .3, .7]).reshape(-1,1)
print(new_data)

predictions = sbs.predict(new_data)
print(predictions)

# Save a checkpoint
#
cp_path = 'PyTorchStepByStep/PaulsStepByStep/model_checkpoint.pth'
sbs.save_checkpoint(cp_path)

# Now resume training with a blank model.
#
model, optimizer, loss_fn = mc.RunModelConfiguration()

new_sbs = StepByStep.StepByStep(model, loss_fn, optimizer)
new_sbs.load_checkpoint(cp_path)
new_sbs.set_loaders(train_loader, val_loader)
new_sbs.train(n_epochs=50)

new_sbs.plot_losses()

print(sbs.model.state_dict())

i = 0