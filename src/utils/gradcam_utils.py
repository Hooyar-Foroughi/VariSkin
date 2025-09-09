import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


# Grad-CAM Implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook to get gradients
        self.target_layer.register_full_backward_hook(self.save_gradients)
        # Hook to get activations
        self.target_layer.register_forward_hook(self.save_activations)

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def save_activations(self, module, input, output):
        self.activations = output

    def generate_cam(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        # If the output is a tuple (regression_output, spatial_map), take the regression output.
        if isinstance(output, tuple):
            output = output[0]

        if class_idx is None:
            # For regression, there's only one output so class_idx is 0.
            class_idx = 0

        score = output[:, class_idx]
        score.backward()

        gradients = self.gradients.cpu().data.numpy()
        activations = self.activations.cpu().data.numpy()

        pooled_gradients = np.mean(gradients, axis=(2, 3))
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[:, i, None, None]

        heatmap = np.mean(activations, axis=1)
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        else:
            print("Warning: Heatmap max value is zero, skipping normalization.")
            heatmap = np.zeros_like(heatmap)

        return heatmap

    def overlay_heatmap(self, heatmap, image_path, pred_score, gt_score):
        # Ensure output directory exists
        output_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "analysis", "gradcam_results")
        )
        os.makedirs(output_dir, exist_ok=True)  # Create folder if it doesn't exist

        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image at {image_path}")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        heatmap = cv2.resize(heatmap[0], (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)

        # Prepare text with predicted and ground truth dice score
        text = f"Pred: {pred_score:.2f}"
        if gt_score is not None:
            text += f" | GT: {gt_score:.2f}"

        # Overlay the text on the image (adjust position, font, and color as needed)
        cv2.putText(
            overlay,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Get image filename
        image_filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"gradcam_{image_filename}")

        # Save image
        cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Grad-CAM saved: {output_path}")


# Selecting target layer for different models
def get_target_layer(model_name, model):
    if model_name == "resnet18":
        # For our multi-task model, the head is at index 1 and the conv layer is inside the head.
        return model[1].conv
    elif model_name == "efficientnet_b0":
        return model.features[-1]
    elif model_name == "vit_b_32":
        return model.encoder.layers[-1]  # May require adaptation for ViT
    else:
        raise ValueError("Model not recognized for Grad-CAM.")


# Example usage
def run_gradcam(model_name, model, image_path, metrics_csv_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    target_layer = get_target_layer(model_name, model)
    gradcam = GradCAM(model, target_layer)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Forward pass to get the predicted Dice score
    output = model(input_tensor)
    if isinstance(output, tuple):
        output = output[0]
    pred_score = output.item()

    # Retrieve ground truth from CSV if provided
    gt_score = None
    if metrics_csv_path is not None:
        import pandas as pd

        df = pd.read_csv(metrics_csv_path)
        img_filename = os.path.basename(image_path)
        row = df[df["img_name"] == img_filename]
        if not row.empty:
            gt_score = row["dc_mean"].values[0]

    # Generate heatmap and overlay it with scores

    heatmap = gradcam.generate_cam(input_tensor)
    gradcam.overlay_heatmap(heatmap, image_path, pred_score, gt_score)
