from datasets import load_dataset
from torchvision import transforms
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, ConcatDataset

def default_transform():
    return Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
    ])

# define function
def transforms(examples, transform=default_transform()):
   examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
   del examples["image"]

   return examples


def load_fashion_mnist(batch_size=128):
    # load dataset from the hub
    dataset = load_dataset("fashion_mnist")
    # Keep labels so we can apply classifier-free guidance
    transformed_dataset = dataset.with_transform(transforms)

    # create dataloader
    dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)

    # Clear memory
    dataset = None
    transformed_dataset = None

    return dataloader

def load_xray():
    # load dataset from the hub
    dataset = load_dataset("keremberke/chest-xray-classification", "full")
    batch_size = 128
    # Keep labels so we can apply classifier-free guidance
    transformed_dataset = dataset.with_transform(transforms)

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
