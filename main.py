import torch
import numpy as np
import neptune.new as neptune
from dset import CatsAnDogs
import models
from preproc import preprocData
from train import train

if __name__ == '__main__':
    neptune_run = neptune.init(
        project='andrekh/catsandogs',
        source_files=['*.py'],
        api_token='')  # your credentials

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    torch.backends.cudnn.benchmark = True

    # PARAMETERS
    batch_size = 128
    num_workers = 2
    max_epochs = 100
    validation_split = 0.2
    random_seed = 12
    dim = 128
    learning_rate = 1e-4
    np.random.seed(random_seed)

    # INSTANTIATE MODEL
    model = models.modelB_dim128.Net(dim)
    model.to(device)
    # number of params
    print("Number of parameters: ", model.getTrainableParameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    params = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'validation_split': validation_split,
        'dim': dim,
        'learning_rate': learning_rate,
        'optimizer': str(type(optimizer).__name__),
        'number_of_parameters': model.getTrainableParameters(),
        'model_architecture': model.getModelName()
    }

    neptune_run['parameters'] = params

    # pre-process data
    data_path = "./catsdogs/train/"
    partition, labels = preprocData(data_path, validation_split)

    # generators
    training_ds = CatsAnDogs(partition['train'], labels, dim)
    training_generator = torch.utils.data.DataLoader(training_ds, batch_size=batch_size, num_workers=num_workers,
                                                     shuffle=True)

    validation_ds = CatsAnDogs(partition['validation'], labels, dim)
    validation_generator = torch.utils.data.DataLoader(validation_ds, batch_size=batch_size, num_workers=num_workers,
                                                       shuffle=True)

    # training
    print("Starting training...")
    train(model, max_epochs, optimizer, training_generator, validation_generator, device, neptune_run)
