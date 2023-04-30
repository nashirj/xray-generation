"""This code originally from my assignment 1, CAP 5516"""
import os

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from torch.utils.data import WeightedRandomSampler

from src import utils


def original_scale_default_transforms():
    """Used to load original size data for classifiers.

    CURRENTLY UNUSED FOR THIS PROJECT
    
    Returns: dictionary of transforms for the train, val, and test sets.
    """
    return {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ]),
    }


# Used as baseline for real original size training data for classifiers
def original_scale_normalized_transforms(mean, std):
    """Used to load original size normalized data for classifiers.

    CURRENTLY UNUSED FOR THIS PROJECT
    
    Returns: dictionary of transforms for the train, val, and test sets.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    return data_transforms


def original_scale_normalized_augmented_transforms(mean, std):
    """Used to augment real original size training data for classifiers.
    
    NOT USED FOR THIS PROJECT
    """
    data_transforms = {
        # Data augmentation and normalization for training
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        # Just normalization for validation
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    return data_transforms


def downscaled_default_diffusion_transforms():
    """Used for already downscaled synthetic data for classifiers.
    
    ResNet expects RGB.

    Returns: dictionary of transforms for the train, and val sets.

    Note: there is no test set for the diffusion data as we use real data for testing.
    """
    return {
        'train': transforms.Compose([
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            transforms.ToTensor()
        ]),
    }


def downscaled_normalized_diffusion_transforms(mean, std):
    """Used for already downscaled synthetic data for classifiers.

    ResNet expects RGB.

    Returns: dictionary of transforms for the train, and val sets.

    Note: there is no test set for the diffusion data as we use real data for testing.
    """
    return {
        'train': transforms.Compose([
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }



def downscaled_normalized_augmented_diffusion_transforms(mean, std):
    """Used for already downscaled synthetic data for classifiers.

    ResNet expects RGB, so we repeat the grayscale image 3 times.

    Returns: dictionary of transforms for the train, and val sets.

    Note: there is no test set for the diffusion data as we use real data for testing.
    """
    return {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1)),
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }


def downscaling_default_transforms(load_as_rgb=True):
    """Used to downscale real xray data for both classifiers and diffusion models."""
    if load_as_rgb:
        return {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize(28),
                transforms.ToTensor()
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize(28),
                transforms.ToTensor()
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize(28),
                transforms.ToTensor()
            ]),
        }
    return {
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Resize(28),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Resize(28),
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Resize(28),
            transforms.ToTensor()
        ]),
    }


def downscaling_normalized_transforms(mean, std, load_as_rgb=True):
    """Used to downscale and normalize real xray data used for training classifiers or diffusion models.
    
    Loads as grayscale by default.
    """
    if load_as_rgb:
        return {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize(28),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize(28),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize(28),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
        }
    return {
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }


def downscaling_normalized_augmented_transforms(mean, std):
    """Used to normalize and downscale real xray data used for training classifiers.
    
    Classifiers use ResNet which expects RGB.
    """
    return {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1)),
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }


def load_oversampled_real_xray_data(data_dir, data_transforms, batch_size=4):
    """Create a balanced dataset by oversampling the minority classes."""
    initial_train_set = datasets.ImageFolder(
        os.path.join(data_dir, 'train'), data_transforms['train'])
    class_names = initial_train_set.classes
    initial_val_set = datasets.ImageFolder(
        os.path.join(data_dir, 'val'), data_transforms['val'])
    test_set = datasets.ImageFolder(
        os.path.join(data_dir, 'test'), data_transforms['val'])

    # Redistribute training/validation samples to increase the size of the validation set
    # such that 80% of the data is used for training and 20% for validation
    all_train_data = torch.utils.dataDown.ConcatDataset([initial_train_set, initial_val_set])
    train_size = int(0.8 * len(all_train_data))
    val_size = len(all_train_data) - train_size
    train_data, val_data = torch.utils.data.random_split(all_train_data, [train_size, val_size])

    # Oversample the minority classes in the training set
    # to create a balanced dataset
    train_labels = [s[1] for s in train_data]
    _, counts = torch.unique(torch.tensor(train_labels), return_counts=True)
    class_weights = [sum(counts) / c for c in counts]
    weight = [class_weights[t] for t in train_labels]
    sampler = WeightedRandomSampler(weight, len(train_labels))
    
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, num_workers=4, sampler=sampler)
    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=4)
    
    dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    dataset_sizes = {'train': len(train_data), 'val': len(val_data), 'test': len(test_set)}

    return dataloaders, dataset_sizes, class_names


def load_real_xray_data(data_dir, data_transforms, batch_size=1, return_val_set=True):
    initial_train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
    initial_val_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
    # Redistribute training/validation samples to increase the size of the validation set
    # such that 80% of the data is used for training and 20% for validation
    all_train_data = torch.utils.data.ConcatDataset([initial_train_set, initial_val_set])
    test_set = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])

    if return_val_set:
        train_size = int(0.8 * len(all_train_data))
        val_size = len(all_train_data) - train_size
        train_set, val_set = torch.utils.data.random_split(all_train_data, [train_size, val_size])

        image_datasets = {'train': train_set, 'val': val_set, 'test': test_set}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                    shuffle=True, num_workers=4)
                                                    for x in ['train', 'val', 'test']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    else:
        image_datasets = {'train': all_train_data, 'test': test_set}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                    shuffle=True, num_workers=4)
                                                    for x in ['train', 'test']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    class_names = initial_train_set.classes

    return dataloaders, dataset_sizes, class_names


def load_diffusion_xray_data(diff_dir, data_dir, diff_transform, real_transforms, batch_size=4):
    """Load a training data consisting of diffusion only data, but make validation and test sets based on real data."""
    # Use diffusion data for the train set
    diff_train_data = datasets.ImageFolder(os.path.join(diff_dir, 'train'), diff_transform['train'])

    # Uses real data for the val and test set
    # Sample 20% of the real data for the validation set
    real_train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), real_transforms['val'])
    real_val_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), real_transforms['val'])
    real_train_data = torch.utils.data.ConcatDataset([real_train_set, real_val_set])
    real_train_size = int(0.8 * len(real_train_data))
    real_val_size = len(real_train_data) - real_train_size
    _, real_val_set = torch.utils.data.random_split(real_train_data, [real_train_size, real_val_size])

    class_names = diff_train_data.classes
    real_test_set = datasets.ImageFolder(os.path.join(data_dir, 'test'), real_transforms['test'])

    image_datasets = {'train': diff_train_data, 'val': real_val_set, 'test': real_test_set}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=4)
                                                for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    return dataloaders, dataset_sizes, class_names


def load_diffusion_normalized_xray_data(diff_dir, data_dir, with_augmentation=False, batch_size=8):
    """Load diffusion + real data for clf; normalize using diffusion data mean and std."""
    diff_transform = downscaled_default_diffusion_transforms()
    real_transform = downscaling_default_transforms(load_as_rgb=True)
    dataloaders, _, _ = load_diffusion_xray_data(diff_dir, data_dir, diff_transform,
                                                 real_transform, batch_size)

    # Compute mean and std of the diffusion data
    mean, std = compute_mean_and_std(dataloaders['train'])

    # Reload downscaled dataset with mean and std computed above
    if with_augmentation:
        diff_transform = downscaled_normalized_augmented_diffusion_transforms(mean, std)
    else:
        diff_transform = downscaled_normalized_diffusion_transforms(mean, std)
    
    real_transform = downscaling_normalized_transforms(mean, std, load_as_rgb=True)

    return load_diffusion_xray_data(diff_dir, data_dir, diff_transform, real_transform, batch_size)


def load_real_downscaled_normalized_xray_data(data_dir, with_augmentation=False, batch_size=8,
                                              return_val_set=False, load_as_rgb=False):
    transform = downscaling_default_transforms(load_as_rgb=load_as_rgb)
    dataloaders, _, _ = load_real_xray_data(data_dir, data_transforms=transform,
                                            return_val_set=return_val_set, batch_size=batch_size)
    mean, std = compute_mean_and_std(dataloaders['train'])
    # Reload downscaled dataset with mean and std computed above
    if with_augmentation:
        transform = downscaling_normalized_augmented_transforms(mean, std, load_as_rgb=load_as_rgb)
    else:
        transform = downscaling_normalized_transforms(mean, std, load_as_rgb=load_as_rgb)
    return load_real_xray_data(data_dir, transform, batch_size=batch_size, return_val_set=return_val_set)


def get_labels(dataloader):
    """Get all labels from a dataloader. Used to compute class weights, which in turn is used
    for oversampling."""
    labels = []
    for _, label in dataloader:
        labels.append(label)
    # Flatten list of lists
    return utils.flatten_list(labels)


def compute_class_weights(labels):
    # Count number of samples of each class
    num_normal = labels.count(0)
    num_pneumonia = labels.count(1)

    weight_for_0 = num_pneumonia / (num_normal + num_pneumonia)
    weight_for_1 = num_normal / (num_normal + num_pneumonia)
    class_weights = [weight_for_0, weight_for_1]
    return torch.FloatTensor(class_weights)


def compute_mean_and_std(dataloader):
    # Compute approximate mean and std of train dataset based on a single batch
    images, _ = next(iter(dataloader))
    # shape of images = [b,c,w,h]
    mean, std = images.mean([0,2,3]), images.std([0,2,3])
    return mean, std



# if __name__ == '__main__':
#     dataloaders, dataset_sizes, class_names = load_oversampled_xray_data()
#     print('Train size: ', dataset_sizes['train'])
#     print('Val size: ', dataset_sizes['val'])
#     print('Test size: ', dataset_sizes['test'])
#     print('Class names: ', class_names)

#     # Get number of samples of each class in train dataloader
#     train_labels = [s[1] for s in dataloaders['train'].dataset]
#     _, counts = torch.unique(torch.tensor(train_labels), return_counts=True)
#     print('Train class counts: ', counts)
