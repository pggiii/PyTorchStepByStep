import DataGenerationV1 as dg
import DataPreparationV0 as dp
import ModelConfigurationV0 as mc
import ModelTrainingV0 as mt

x_train, y_train, x_val, y_val = dg.RunDataGeneration()

device, x_train_tensor, y_train_tensor =  dp.RunDataPreparation(x_train, y_train)

device, model, optimizer, loss_fn = mc.RunModelConfiguration(x_train_tensor, y_train_tensor)

mt.RunModelTraining(x_train_tensor, y_train_tensor, model, optimizer, loss_fn)

i = 0