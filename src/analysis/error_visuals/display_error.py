import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch


def display_extreme_error_images(model, dataset, device, num_images=5):
    # Computes the absolute error for each test image, selects the lowest and highest error images,
    # and displays them in a 2-row plot. In addition, prints the error, ground truth, and prediction
    # for each selected image.
    model.eval()
    errors = []
    image_paths = []
    gt_values = []
    pred_values = []

    # Loop through the test dataset to compute predictions and errors.
    with torch.no_grad():
        for i in range(len(dataset)):
            # Get the transformed image and ground truth label.
            image, label = dataset[i]
            image_batch = image.unsqueeze(0).to(device)  # add batch dimension

            # Get the model prediction; model returns a tuple (output, spatial_map)
            output = model(image_batch)
            if isinstance(output, tuple):
                output = output[0]
            pred = output.item()

            # Calculate absolute error.
            error = abs(pred - label.item())
            errors.append(error)
            gt_values.append(label.item())
            pred_values.append(pred)

            # Retrieve the image path from the dataset.
            img_name = dataset.df.iloc[i]["img_name"]
            img_path = os.path.join(dataset.img_dir, img_name)
            image_paths.append(img_path)

    errors = np.array(errors)

    # Get indices for lowest and highest errors.
    lowest_indices = np.argsort(errors)[:num_images]
    highest_indices = np.argsort(errors)[-num_images:][::-1]  # highest first

    # Print error details to console.
    print("Lowest Error Images:")
    for idx in lowest_indices:
        print(
            f"Image: {image_paths[idx]}, GT: {gt_values[idx]:.2f}, Pred: {pred_values[idx]:.2f}, Error: {errors[idx]:.2f}"
        )

    print("\nHighest Error Images:")
    for idx in highest_indices:
        print(
            f"Image: {image_paths[idx]}, GT: {gt_values[idx]:.2f}, Pred: {pred_values[idx]:.2f}, Error: {errors[idx]:.2f}"
        )

    # Create a plot with two rows: lowest errors on top, highest errors on bottom.
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 3, 6))

    for j, idx in enumerate(lowest_indices):
        img = Image.open(image_paths[idx]).convert("RGB")
        axes[0, j].imshow(img)
        axes[0, j].axis("off")
        axes[0, j].set_title(
            f"Low Err\nGT: {gt_values[idx]:.2f}\nPred: {pred_values[idx]:.2f}\nErr: {errors[idx]:.2f}"
        )

    for j, idx in enumerate(highest_indices):
        img = Image.open(image_paths[idx]).convert("RGB")
        axes[1, j].imshow(img)
        axes[1, j].axis("off")
        axes[1, j].set_title(
            f"High Err\nGT: {gt_values[idx]:.2f}\nPred: {pred_values[idx]:.2f}\nErr: {errors[idx]:.2f}"
        )

    plt.tight_layout()
    plt.show()
    # Save the error visuals for headless environments
    output_path = os.path.join(os.path.dirname(__file__), "error_visuals.png")
    plt.savefig(output_path)
