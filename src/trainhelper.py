import time

import numpy as np
import torch
from torchvision.utils import save_image

from src import viz

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def train_classifier_free_guidance(epochs, model, dataloader, optimizer, device, diffusion,
                                   results_folder, label_map, losses=[], log_every=100,
                                   model_name="model", cond_scale=3.):
    timestamp = int(time.time())
    timestamp = time.strftime('%m-%d-%Y--%H-%M', time.localtime(timestamp))
    print(f"Starting training at {timestamp}")

    best_model_sd = model.state_dict()
    best_loss = float("inf")
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        epoch_losses = []
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            training_images = batch[0].to(device)
            image_classes = batch[1].to(device)

            loss = diffusion(training_images, classes = image_classes)
            epoch_losses.append(loss.item())

            # Save model if loss is lower than previous best
            if loss < best_loss:
                best_model_sd = model.state_dict()
                best_loss = loss

            if step % log_every == 0:
                print(f"Loss at step {step}: {loss.item()}")

            loss.backward()
            optimizer.step()

        losses.append(epoch_losses)
        # After each epoch, save some generated images
        sampled_images, image_classes = sample_n_images(diffusion, 10, 2, cond_scale)
        save_path = results_folder / f"generated-images/{model_name}-{timestamp}-epoch-{epoch}.png"
        viz.plot_generated_images(sampled_images, image_classes, label_map, save_path)

    print("Finished training, saving model and losses")
    # Save model with timestamp
    torch.save(best_model_sd, str(results_folder / f"models/{model_name}-{timestamp}.pt"))
    # Save losses with timestamp
    torch.save(losses, str(results_folder / f"losses/{model_name}-{timestamp}.pt"))

    # Load best model statedict into model
    model.load_state_dict(best_model_sd)

    return losses

def sample_n_images(diffusion, n_imgs, n_classes, cond_scale):
    image_classes = [i for i in range(n_classes)]
    # Uniformly sample from img classes to get n_imgs
    image_classes = np.random.choice(image_classes, n_imgs)
    # convert to tensor
    image_classes = torch.tensor(image_classes).cuda()

    sampled_images = diffusion.sample(
        classes = image_classes,
        cond_scale = cond_scale
    )

    return sampled_images.cpu(), image_classes.cpu().numpy().tolist()
