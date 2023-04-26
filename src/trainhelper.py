import time

import torch
from torchvision.utils import save_image

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def train_classifier_free_guidance(epochs, model, dataloader, optimizer, device, diffusion,
                                   results_folder, save_and_sample_every=100, cond_scale=3.,
                                   losses=[]):
    best_model = None
    best_loss = float("inf")
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            training_images = batch["pixel_values"].to(device)
            image_classes = batch["label"].to(device)

            loss = diffusion(training_images, classes = image_classes)
            losses.append(loss.item())

            # Save model if loss is lower than previous best
            if loss < best_loss:
                best_model = model
                best_loss = loss

            if step % 100 == 0:
                print(f"Loss at step {step}: {loss.item()}")

            loss.backward()
            optimizer.step()

            # # save generated images
            # if step != 0 and step % save_and_sample_every == 0:
            #     milestone = step // save_and_sample_every
            #     batches = num_to_groups(4, batch_size)
            #     all_images_list = list(map(lambda n: diffusion.sample(classes=image_classes, cond_scale=cond_scale), batches))
            #     all_images = torch.cat(all_images_list, dim=0)
            #     all_images = (all_images + 1) * 0.5
            #     save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = 6)

    print("Finished training, saving model")
    timestamp = int(time.time())
    # Save model with timestamp
    torch.save(best_model.state_dict(), str(results_folder / f"model-{timestamp}.pt"))
    # Save losses with timestamp
    torch.save(losses, str(results_folder / f"losses-{timestamp}.pt"))
