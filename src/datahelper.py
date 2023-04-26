from datasets import load_dataset
from torchvision import transforms
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.datasets as dset

def default_transform():
    return Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
    ])

def torch_transform():
    return Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
    ])

# define function
def ds_transforms(examples, transform=default_transform()):
   examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
   del examples["image"]

   return examples


def load_fashion_mnist(batch_size=128):
    # load dataset from the hub
    dataset = load_dataset("fashion_mnist")
    # Keep labels so we can apply classifier-free guidance
    transformed_dataset = dataset.with_transform(ds_transforms)

    # create dataloader
    dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)

    # Clear memory
    dataset = None
    transformed_dataset = None

    return dataloader

def load_fashion_mnist_from_torch(batch_size=128):
    # load dataset from the hub
    dataset = dset.FashionMNIST(root="data", download=True, transform=torch_transform())

    # create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def load_xray(transform=default_transform()):
    # load dataset from the hub
    dataset = load_dataset("keremberke/chest-xray-classification", "full")
    batch_size = 128
    # Keep labels so we can apply classifier-free guidance
    transformed_dataset = dataset.with_transform(transform)

    train_set = ConcatDataset([
        transformed_dataset["train"],
        transformed_dataset["validation"]
    ])

    # create dataloader
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Hold out some test data for evaluation, FID, KID, etc.
    test_dl = DataLoader(transformed_dataset["test"], batch_size=batch_size, shuffle=True)

    # Clear memory
    dataset = None
    transformed_dataset = None

    return train_dl, test_dl
