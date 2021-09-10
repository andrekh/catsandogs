import glob
import numpy as np
import os


def preprocData(data_path, validation_split):
    training_filenames_list = [x for x in glob.glob(data_path + "*.jpg")]
    indices = [i for i, _ in enumerate(training_filenames_list)]
    np.random.shuffle(indices)
    split = int(np.floor((1 - validation_split) * len(indices)))
    train_indices, val_indices = indices[:split], indices[split:]

    # label: 0 for cat, 1 for dog
    partition = {'train': [], 'validation': []}
    labels = {}
    for idx in train_indices:
        partition['train'].append(training_filenames_list[idx])
    for idx in val_indices:
        partition['validation'].append(training_filenames_list[idx])
    for idx in indices:
        labels[training_filenames_list[idx]] = 0 if \
            os.path.splitext(os.path.basename(training_filenames_list[idx]))[
                0].startswith('cat') else 1

    return (partition, labels)
