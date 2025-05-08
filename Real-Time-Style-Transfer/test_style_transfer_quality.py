import torch
import argparse
import os
import time
import csv
from PIL import Image
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
import numpy as np
import lpips
from tqdm import tqdm

def load_image(image_path, size=(256, 256)):
    """Load and preprocess an image."""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),  # Resize to match C++ pipeline (256x256)
        transforms.ToTensor(),    # Converts to CHW, [0, 1]
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
    return image

def convert_tensor_to_numpy(tensor):
    """Convert normalized tensor to numpy array for SSIM calculation."""
    # Remove batch dimension and denormalize
    tensor = tensor.squeeze(0).cpu()  # 3xHxW
    denorm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                  std=[1/0.229, 1/0.224, 1/0.225])
    tensor = denorm(tensor)  # Back to [0, 1]
    tensor = tensor.clamp(0, 1)
    # Convert to numpy and transpose from CHW to HWC
    np_img = tensor.permute(1, 2, 0).numpy()
    return np_img

def calculate_ssim(img1, img2):
    """Calculate SSIM between two numpy images (HWC format)."""
    # Calculate SSIM for each channel and take the mean
    ssim_value = ssim(img1, img2, data_range=1.0, channel_axis=2, win_size=11)
    return ssim_value

def load_and_prepare_lpips():
    """Load LPIPS model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net='alex').to(device)
    return loss_fn

def calculate_lpips(img1_tensor, img2_tensor, loss_fn):
    """Calculate LPIPS between two tensor images."""
    # LPIPS expects tensors in range [-1, 1]
    # Our tensors are already normalized for the model, but need conversion for LPIPS
    # First denormalize to [0, 1]
    denorm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                 std=[1/0.229, 1/0.224, 1/0.225])
    img1_denorm = denorm(img1_tensor.squeeze(0))
    img2_denorm = denorm(img2_tensor.squeeze(0))
    
    # Then normalize to [-1, 1] for LPIPS
    img1_lpips = img1_denorm * 2 - 1
    img2_lpips = img2_denorm * 2 - 1
    
    # Add batch dimension back
    img1_lpips = img1_lpips.unsqueeze(0)
    img2_lpips = img2_lpips.unsqueeze(0)
    
    # Calculate LPIPS
    with torch.no_grad():
        lpips_value = loss_fn(img1_lpips, img2_lpips)
    
    return lpips_value.item()

def evaluate_dataset(model_path, dataset_path, output_dir, num_samples=None):
    """Evaluate style transfer on a dataset and compute metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load LPIPS model
    lpips_fn = load_and_prepare_lpips()
    
    # Get list of image files
    image_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    # Use a subset of images if specified
    if num_samples and num_samples < len(image_files):
        image_files = image_files[:num_samples]
    
    print(f"Found {len(image_files)} images for evaluation")
    
    # Prepare CSV file for results
    csv_path = os.path.join(output_dir, "metrics_results.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Image', 'SSIM', 'LPIPS', 'Processing Time (s)'])
    
    # Metrics storage
    ssim_scores = []
    lpips_scores = []
    processing_times = []
    
    # Process each image
    for i, img_path in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            # Load and preprocess the image
            image = load_image(img_path)
            image = image.to(device)
            
            # Measure processing time
            start_time = time.time()
            
            # Run inference
            with torch.no_grad():
                stylized = model(image)
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Convert to numpy for SSIM calculation
            original_np = convert_tensor_to_numpy(image)
            stylized_np = convert_tensor_to_numpy(stylized)
            
            # Calculate SSIM
            ssim_value = calculate_ssim(original_np, stylized_np)
            ssim_scores.append(ssim_value)
            
            # Calculate LPIPS
            lpips_value = calculate_lpips(image, stylized, lpips_fn)
            lpips_scores.append(lpips_value)
            
            # Save a sample of stylized images (e.g., every 100th image)
            if i % 100 == 0:
                img_name = os.path.basename(img_path)
                output_path = os.path.join(output_dir, f"stylized_{img_name}")
                save_image(stylized.cpu(), output_path)
                
                # For comparison, also save the original
                original_output_path = os.path.join(output_dir, f"original_{img_name}")
                save_image(image.cpu(), original_output_path)
            
            # Write results to CSV
            with open(csv_path, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([img_path, ssim_value, lpips_value, processing_time])
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Calculate summary statistics
    avg_ssim = np.mean(ssim_scores)
    avg_lpips = np.mean(lpips_scores)
    avg_processing_time = np.mean(processing_times)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Average SSIM: {avg_ssim:.4f} (higher is better)")
    print(f"Average LPIPS: {avg_lpips:.4f} (lower is better)")
    print(f"Average processing time: {avg_processing_time:.4f} seconds per image")
    
    # Write summary to file
    with open(os.path.join(output_dir, "summary.txt"), 'w') as f:
        f.write("Evaluation Summary:\n")
        f.write(f"Total images evaluated: {len(image_files)}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f} (higher is better)\n")
        f.write(f"Average LPIPS: {avg_lpips:.4f} (lower is better)\n")
        f.write(f"Average processing time: {avg_processing_time:.4f} seconds per image\n")
    
    # Save the raw scores for later visualization if needed
    np.save(os.path.join(output_dir, "ssim_scores.npy"), np.array(ssim_scores))
    np.save(os.path.join(output_dir, "lpips_scores.npy"), np.array(lpips_scores))
    print(f"Saved raw scores to {output_dir} for later visualization")
    
    return {
        'avg_ssim': avg_ssim,
        'avg_lpips': avg_lpips,
        'avg_processing_time': avg_processing_time
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate style transfer model on a dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.pth file")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save results")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate (default: all)")
    
    args = parser.parse_args()
    
    evaluate_dataset(args.model_path, args.dataset_path, args.output_dir, args.num_samples)

if __name__ == "__main__":
    main()
