from torchvision.utils import save_image

from src import dataset
from src import datahelper as dh


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def average_nested_lists(l):
    return [sum(x) / len(x) for x in l]


def save_n_images_from_dataloader(dataloader, n, save_dir, image_key=0, label_key=1):
    # Save n images from the provided dataloader to specified path
    n_saved = 0
    for i, batch in enumerate(dataloader):
        # Save each image in batch
        for j, (image, label) in enumerate(zip(batch[image_key], batch[label_key])):
            save_image(image, f"{save_dir}/{label}-{i}-{j}.png")
            n_saved += 1
            if n_saved >= n:
                return


if __name__ == "__main__":
    # Save fashion mnist images
    dl = dh.load_fashion_mnist()
    save_n_images_from_dataloader(dl, 100, "data/metric-comparison/fmnist/real", image_key="pixel_values", label_key="label")


    # # Save real images
    # dl, sizes, classes = dataset.load_downscaled_xray_data("../data/chest_xray")

    # train_dl = dl["train"]
    # test_dl = dl["test"]

    # save_n_images_from_dataloader(train_dl, 100, "../data/metric-comparison/xray/real-train")
    # save_n_images_from_dataloader(test_dl, 100, "../data/metric-comparison/xray/real-test")

