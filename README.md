# Medical Image Synthesis for Data Augmentation and Anonymization via Diffusion

This project investigates the use of diffusion models to generate high-fidelity medical imaging data. The generated data is evaluated using image quality metrics (FID and SSIM) and is used as part of a downstream classification task to evaluate the impact synthetic pretraining has on classifier performance.

See the [xray-diffusion notebook](notebooks/xray-diffusion.ipynb) for the details regarding training the diffusion model, and the [classifiers notebook](notebooks/classifiers.ipynb) for details about the classification task.

A [final report](results/final-report.pdf) is also included in the results folder.
