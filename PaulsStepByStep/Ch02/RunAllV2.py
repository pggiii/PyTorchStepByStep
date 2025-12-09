import DataGenerationV1 as dg
import DataPreparationV2 as dp
import ModelConfigurationV3 as mc
import ModelTrainingV5 as mt
import HelpersV0 as helpers

x, y = dg.RunDataGeneration()

train_loader, val_loader =  dp.RunDataPreparation(x, y)

device, train_step_fn, val_step_fn, get_model_fn, writer = mc.RunModelConfiguration(train_loader)

losses, val_losses = mt.RunModelTraining(device, train_loader, val_loader, train_step_fn, val_step_fn, get_model_fn,)

helpers.plot_losses(losses, val_losses)
i = 0