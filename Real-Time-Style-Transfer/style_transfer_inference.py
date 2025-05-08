import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms

def load_image(image_path, size=(256, 256)):
    """Load and preprocess an image."""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),  # Resize to match C++ pipeline (256x256)
        transforms.ToTensor(),  # Converts to CHW, [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Match training normalization
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension: 1x3xHxW
    return image

def save_image(tensor, output_path="output.png"):
    """Save a tensor as an image."""
    # Remove batch dimension and denormalize
    tensor = tensor.squeeze(0)  # 3xHxW
    denorm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                                 std=[1/0.229, 1/0.224, 1/0.225])
    tensor = denorm(tensor)  # Back to [0, 1]
    tensor = tensor.clamp(0, 1)
    
    # Convert to PIL Image and save
    to_pil = transforms.ToPILImage()
    image = to_pil(tensor)
    image.save(output_path)
    print(f"Saved stylized image to {output_path}")

def style_transfer(model_path, image_path):
    """Run style transfer on an image using the given model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load and preprocess the image
    image = load_image(image_path)
    image = image.to(device)
    print(f"Loaded image from {image_path}, shape: {image.shape}")

    # Run inference
    with torch.no_grad():
        try:
            stylized = model(image)
            print(f"Stylized output shape: {stylized.shape}")
        except Exception as e:
            print(f"Error during inference: {e}")
            return

    # Save the result
    stylized = stylized.cpu()
    save_image(stylized)

def main():
    parser = argparse.ArgumentParser(description="Test style transfer on an image.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.pth file")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    style_transfer(args.model_path, args.image_path)

if __name__ == "__main__":
    main()
