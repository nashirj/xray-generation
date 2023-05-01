# Notes
When constructing the Unet, need the number of resnert blocks to be divisible by the output dimension. When copying the code from the [lucidrains repo](https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py), I had to change the default number of resnet_blocks from 8 to 4 so that the size of fashion mnist images (28x28) was divisible by the number of groups on line 280.


xray dataset from: https://huggingface.co/datasets/keremberke/chest-xray-classification
fashion mnist from: https://huggingface.co/datasets/fashion_mnist


generative-adversarial-network.png source: https://www.kdnuggets.com/2017/01/generative-adversarial-networks-hot-topic-machine-learning.html


# Diagrams
https://www.figma.com/file/XhWmmWcprQTY6qwCUSqa54/Diagramming-101-(Community)-(Copy)?node-id=0-1&t=ANHMtjONbsxHJiRp-0




TODOs:
- clean repo
- train diffusion model for longer
- generate more images to train classifier
- create directories for each label of generated image to make building training set more straightforward
- MAKE SURE NO TEST DATA IS SEEN DURING DIFFUSION TRAINING



# notes
want to make it clear that diffusion beats traditional data augmentation

also make the point about diffusion being useful for addressing class imbalance problem
quantify the comparison of normal balancing (oversampling, class weighting) vs. diffusion data balancing

add SSIM

try to compare to others if possible, conditional gan, conditional vae

include tables
include different data splits, fully synthetic vs all normal vs split, for varying numbers of samples

train on general synthetic dataset, then fine tune on local patient data on a per institution basis