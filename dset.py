import torch
import torchvision.transforms as T
from PIL import Image


class CatsAnDogs(torch.utils.data.Dataset):
    def __init__(self, list_IDs, labels, dim):
        self.labels = labels
        self.list_IDs = list_IDs
        self.dim = dim
        self.preprocess = T.Compose([
            T.Resize((dim, dim)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        img = Image.open(ID)
        label = self.labels[ID]
        return self.preprocess(img), label
