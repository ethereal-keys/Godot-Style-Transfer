import torch
import os
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

import utils

from network import normal
from network import slim
from dataset import IDataset
from vgg import Vgg16


def trainer(args):
    print_args(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    imgs_path = args.dataset
    test_imgs_path = args.test_dataset
    style_image = args.style_image

    size = (args.image_width, args.image_height)

    # Define network
    if args.model_mode != "slim":
        image_transformer = normal.ImageTransformNet().to(device)
    else:
        # Note: slim.py has been modified to remove input_size dependency
        image_transformer = slim.ImageTransformNet().to(device)

    # Set optimizer
    optimizer = Adam(image_transformer.parameters(), args.learning_rate)

    # Define loss function
    loss_mse = nn.MSELoss()

    # Load VGG network
    vgg = Vgg16(args.VGG_path).to(device)

    # Get training dataset
    dataset_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode == 'RGBA' else img),
        transforms.Resize(size),
        transforms.ToTensor(),
        utils.normalize_tensor_transform()
    ])
    train_dataset = IDataset(imgs_path, dataset_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    # Get testing dataset
    test_dataset = IDataset(test_imgs_path, dataset_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # Style image
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        utils.normalize_tensor_transform()
    ])
    style = utils.load_image(style_image)
    style = style_transform(style)
    style = Variable(style.repeat(args.batch_size, 1, 1, 1)).to(device)
    style_name = os.path.split(style_image)[-1].split('.')[0]

    # Calculate gram matrices for style features
    style_features = vgg(style)
    style_gram = [utils.gram(fmap) for fmap in style_features]

    # Create lists to store loss values for plotting
    train_style_losses = []
    train_content_losses = []
    train_tv_losses = []
    train_total_losses = []
    
    test_style_losses = []
    test_content_losses = []
    test_tv_losses = []
    test_total_losses = []
    
    # To store average losses per epoch
    epoch_train_style_losses = []
    epoch_train_content_losses = []
    epoch_train_tv_losses = []
    epoch_train_total_losses = []
    
    epoch_test_style_losses = []
    epoch_test_content_losses = []
    epoch_test_tv_losses = []
    epoch_test_total_losses = []

    print("Start training. . .")
    best_style_loss = 1e9
    best_content_loss = 1e9
    for e in range(args.epoch):
        # Training phase
        img_count = 0
        aggregate_style_loss = 0.0
        aggregate_content_loss = 0.0
        aggregate_tv_loss = 0.0
        aggregate_total_loss = 0.0

        image_transformer.train()
        for batch_num, x in enumerate(train_loader):
            img_batch_read = len(x)
            img_count += img_batch_read

            optimizer.zero_grad()

            x = Variable(x).to(device)
            y_hat = image_transformer(x)

            y_c_features = vgg(x)
            y_hat_features = vgg(y_hat)

            # Style loss
            y_hat_gram = [utils.gram(fmap) for fmap in y_hat_features]
            style_loss = 0.0
            for j in range(4):
                style_loss += loss_mse(y_hat_gram[j], style_gram[j][:img_batch_read])
            style_loss = args.style_weight * style_loss
            aggregate_style_loss += style_loss.item()

            # Content loss
            recon = y_c_features[1]
            recon_hat = y_hat_features[1]
            content_loss = args.content_weight * loss_mse(recon_hat, recon)
            aggregate_content_loss += content_loss.item()

            # Total variation loss
            diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
            diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
            tv_loss = args.tv_weight * (diff_i + diff_j)
            aggregate_tv_loss += tv_loss.item()

            total_loss = style_loss + content_loss + tv_loss
            aggregate_total_loss += total_loss.item()

            # Store losses for plotting
            train_style_losses.append(style_loss.item())
            train_content_losses.append(content_loss.item())
            train_tv_losses.append(tv_loss.item())
            train_total_losses.append(total_loss.item())

            total_loss.backward()
            optimizer.step()

            if (batch_num + 1) % 50 == 0:
                status = "{}  Epoch {}:  [{}/{}]  Batch:[{}]  agg_style: {:.6f}  agg_content: {:.6f}  " \
                         "agg_tv: {:.6f}  style: {:.6f}  content: {:.6f}  tv: {:.6f} "\
                    .format(
                            time.ctime(), e + 1, img_count, len(train_dataset), batch_num+1,
                            aggregate_style_loss/(batch_num+1.0),
                            aggregate_content_loss/(batch_num+1.0),
                            aggregate_tv_loss/(batch_num+1.0), style_loss.item(),
                            content_loss.item(), tv_loss.item())
                print(status)

        # Calculate average training losses for this epoch
        num_train_batches = len(train_loader)
        epoch_train_style_loss = aggregate_style_loss / num_train_batches
        epoch_train_content_loss = aggregate_content_loss / num_train_batches
        epoch_train_tv_loss = aggregate_tv_loss / num_train_batches
        epoch_train_total_loss = aggregate_total_loss / num_train_batches
        
        # Store epoch averages for training
        epoch_train_style_losses.append(epoch_train_style_loss)
        epoch_train_content_losses.append(epoch_train_content_loss)
        epoch_train_tv_losses.append(epoch_train_tv_loss)
        epoch_train_total_losses.append(epoch_train_total_loss)

        # Testing phase
        print("Evaluating on test dataset...")
        image_transformer.eval()
        
        test_agg_style_loss = 0.0
        test_agg_content_loss = 0.0
        test_agg_tv_loss = 0.0
        test_agg_total_loss = 0.0
        
        with torch.no_grad():
            for batch_num, x in enumerate(test_loader):
                img_batch_read = len(x)
                
                x = x.to(device)
                y_hat = image_transformer(x)
                
                y_c_features = vgg(x)
                y_hat_features = vgg(y_hat)
                
                # Style loss
                y_hat_gram = [utils.gram(fmap) for fmap in y_hat_features]
                test_style_loss = 0.0
                for j in range(4):
                    test_style_loss += loss_mse(y_hat_gram[j], style_gram[j][:img_batch_read])
                test_style_loss = args.style_weight * test_style_loss
                test_agg_style_loss += test_style_loss.item()
                
                # Content loss
                recon = y_c_features[1]
                recon_hat = y_hat_features[1]
                test_content_loss = args.content_weight * loss_mse(recon_hat, recon)
                test_agg_content_loss += test_content_loss.item()
                
                # Total variation loss
                diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
                diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
                test_tv_loss = args.tv_weight * (diff_i + diff_j)
                test_agg_tv_loss += test_tv_loss.item()
                
                test_total_loss = test_style_loss + test_content_loss + test_tv_loss
                test_agg_total_loss += test_total_loss.item()
                
                # Store individual test losses for plotting
                test_style_losses.append(test_style_loss.item())
                test_content_losses.append(test_content_loss.item())
                test_tv_losses.append(test_tv_loss.item())
                test_total_losses.append(test_total_loss.item())
        
        # Calculate average test losses for this epoch
        num_test_batches = len(test_loader)
        epoch_test_style_loss = test_agg_style_loss / num_test_batches
        epoch_test_content_loss = test_agg_content_loss / num_test_batches
        epoch_test_tv_loss = test_agg_tv_loss / num_test_batches
        epoch_test_total_loss = test_agg_total_loss / num_test_batches
        
        # Store epoch averages for testing
        epoch_test_style_losses.append(epoch_test_style_loss)
        epoch_test_content_losses.append(epoch_test_content_loss)
        epoch_test_tv_losses.append(epoch_test_tv_loss)
        epoch_test_total_losses.append(epoch_test_total_loss)
        
        print("Epoch {} - Train losses: style={:.6f}, content={:.6f}, tv={:.6f}, total={:.6f}".format(
            e+1, epoch_train_style_loss, epoch_train_content_loss, epoch_train_tv_loss, epoch_train_total_loss))
        print("Epoch {} - Test losses: style={:.6f}, content={:.6f}, tv={:.6f}, total={:.6f}".format(
            e+1, epoch_test_style_loss, epoch_test_content_loss, epoch_test_tv_loss, epoch_test_total_loss))

        # Save model as TorchScript
        model_folder = args.model_folder
        if not os.path.exists("models/{}".format(model_folder)):
            os.makedirs("models/{}".format(model_folder))

        filename = "models/{}/{}_epoch={}_style={:.4f}_content={:.4f}_tv={:.4f}.pth".format(
            model_folder, style_name, e + 1, epoch_train_style_loss,
            epoch_train_content_loss, epoch_train_tv_loss)

        if epoch_train_style_loss < best_style_loss or epoch_train_content_loss < best_content_loss:
            # Script the model for dynamic shapes
            print("Scripting the model for dynamic shapes...")
            scripted_model = torch.jit.script(image_transformer)
            scripted_model.save(filename)
            print(f"Saved TorchScript model (scripted) to {filename}")

            # Test the scripted model with different input sizes
            scripted_model = torch.jit.load(filename).to(device)
            test_input_256 = torch.randn(1, 3, 256, 256).to(device)
            test_input_512 = torch.randn(1, 3, 512, 512).to(device)
            with torch.no_grad():
                output_256 = scripted_model(test_input_256)
                output_512 = scripted_model(test_input_512)
            print(f"Scripted model output size for 256x256 input: {output_256.shape}")
            print(f"Scripted model output size for 512x512 input: {output_512.shape}")

        if epoch_train_style_loss < best_style_loss:
            best_style_loss = epoch_train_style_loss
        if epoch_train_content_loss < best_content_loss:
            best_content_loss = epoch_train_content_loss
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join("plots", args.model_folder)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot batch-wise losses
    plot_batch_losses(train_style_losses, train_content_losses, train_tv_losses, train_total_losses,
                     test_style_losses, test_content_losses, test_tv_losses, test_total_losses,
                     plots_dir, style_name, args.model_mode)
    
    # Plot epoch-wise losses
    plot_epoch_losses(epoch_train_style_losses, epoch_train_content_losses, epoch_train_tv_losses, epoch_train_total_losses,
                     epoch_test_style_losses, epoch_test_content_losses, epoch_test_tv_losses, epoch_test_total_losses,
                     plots_dir, style_name, args.model_mode)
    
    # Save loss values to numpy files for later analysis
    save_loss_data(train_style_losses, train_content_losses, train_tv_losses, train_total_losses,
                  test_style_losses, test_content_losses, test_tv_losses, test_total_losses,
                  epoch_train_style_losses, epoch_train_content_losses, epoch_train_tv_losses, epoch_train_total_losses,
                  epoch_test_style_losses, epoch_test_content_losses, epoch_test_tv_losses, epoch_test_total_losses,
                  plots_dir, style_name, args.model_mode)


def plot_batch_losses(train_style_losses, train_content_losses, train_tv_losses, train_total_losses,
                    test_style_losses, test_content_losses, test_tv_losses, test_total_losses,
                    plots_dir, style_name, model_mode):
    """Plot detailed batch-wise loss curves comparing train and test"""
    # Plot style losses
    plt.figure(figsize=(12, 8))
    plt.semilogy(train_style_losses, label='Train Style Loss', alpha=0.7)
    # Use step size to match the length of test data points to train data points
    step = len(train_style_losses) // len(test_style_losses)
    x_test = [i * step for i in range(len(test_style_losses))]
    plt.semilogy(x_test, test_style_losses, label='Test Style Loss', linestyle='--')
    
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.xlabel('Iterations (Batches)')
    plt.ylabel('Style Loss Value (log scale)')
    plt.title(f'Style Loss Comparison - {style_name} - {model_mode} model')
    
    # Save the figure
    loss_plot_path = os.path.join(plots_dir, f'{style_name}_{model_mode}_style_losses.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    print(f"Style loss plot saved to {loss_plot_path}")
    
    # Plot content losses
    plt.figure(figsize=(12, 8))
    plt.semilogy(train_content_losses, label='Train Content Loss', alpha=0.7)
    plt.semilogy(x_test, test_content_losses, label='Test Content Loss', linestyle='--')
    
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.xlabel('Iterations (Batches)')
    plt.ylabel('Content Loss Value (log scale)')
    plt.title(f'Content Loss Comparison - {style_name} - {model_mode} model')
    
    # Save the figure
    loss_plot_path = os.path.join(plots_dir, f'{style_name}_{model_mode}_content_losses.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    print(f"Content loss plot saved to {loss_plot_path}")
    
    # Plot total losses
    plt.figure(figsize=(12, 8))
    plt.semilogy(train_total_losses, label='Train Total Loss', alpha=0.7)
    plt.semilogy(x_test, test_total_losses, label='Test Total Loss', linestyle='--')
    
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.xlabel('Iterations (Batches)')
    plt.ylabel('Total Loss Value (log scale)')
    plt.title(f'Total Loss Comparison - {style_name} - {model_mode} model')
    
    # Save the figure
    loss_plot_path = os.path.join(plots_dir, f'{style_name}_{model_mode}_total_losses.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    print(f"Total loss plot saved to {loss_plot_path}")
    
    # Plot combined losses (all in one)
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.semilogy(train_style_losses, label='Train', alpha=0.7)
    plt.semilogy(x_test, test_style_losses, label='Test', linestyle='--')
    plt.title('Style Loss')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.semilogy(train_content_losses, label='Train', alpha=0.7)
    plt.semilogy(x_test, test_content_losses, label='Test', linestyle='--')
    plt.title('Content Loss')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.semilogy(train_tv_losses, label='Train', alpha=0.7)
    plt.semilogy(x_test, test_tv_losses, label='Test', linestyle='--')
    plt.title('TV Loss')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.semilogy(train_total_losses, label='Train', alpha=0.7)
    plt.semilogy(x_test, test_total_losses, label='Test', linestyle='--')
    plt.title('Total Loss')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend()
    
    plt.suptitle(f'Loss Comparison - {style_name} - {model_mode} model', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    loss_plot_path = os.path.join(plots_dir, f'{style_name}_{model_mode}_combined_losses.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    print(f"Combined loss plot saved to {loss_plot_path}")


def plot_epoch_losses(epoch_train_style_losses, epoch_train_content_losses, epoch_train_tv_losses, epoch_train_total_losses,
                     epoch_test_style_losses, epoch_test_content_losses, epoch_test_tv_losses, epoch_test_total_losses,
                     plots_dir, style_name, model_mode):
    """Plot epoch-wise average loss curves comparing train and test"""
    epochs = range(1, len(epoch_train_style_losses) + 1)
    
    # Style loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, epoch_train_style_losses, 'o-', label='Train Style Loss')
    plt.plot(epochs, epoch_test_style_losses, 's--', label='Test Style Loss')
    
    plt.legend()
    plt.grid(True, ls="--", alpha=0.7)
    plt.xlabel('Epochs')
    plt.ylabel('Average Style Loss Value')
    plt.title(f'Epoch-wise Style Loss - {style_name} - {model_mode} model')
    plt.xticks(epochs)
    
    # Save the figure
    loss_plot_path = os.path.join(plots_dir, f'{style_name}_{model_mode}_epoch_style_loss.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    print(f"Epoch style loss plot saved to {loss_plot_path}")
    
    # Content loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, epoch_train_content_losses, 'o-', label='Train Content Loss')
    plt.plot(epochs, epoch_test_content_losses, 's--', label='Test Content Loss')
    
    plt.legend()
    plt.grid(True, ls="--", alpha=0.7)
    plt.xlabel('Epochs')
    plt.ylabel('Average Content Loss Value')
    plt.title(f'Epoch-wise Content Loss - {style_name} - {model_mode} model')
    plt.xticks(epochs)
    
    # Save the figure
    loss_plot_path = os.path.join(plots_dir, f'{style_name}_{model_mode}_epoch_content_loss.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    print(f"Epoch content loss plot saved to {loss_plot_path}")
    
    # Total loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, epoch_train_total_losses, 'o-', label='Train Total Loss')
    plt.plot(epochs, epoch_test_total_losses, 's--', label='Test Total Loss')
    
    plt.legend()
    plt.grid(True, ls="--", alpha=0.7)
    plt.xlabel('Epochs')
    plt.ylabel('Average Total Loss Value')
    plt.title(f'Epoch-wise Total Loss - {style_name} - {model_mode} model')
    plt.xticks(epochs)
    
    # Save the figure
    loss_plot_path = os.path.join(plots_dir, f'{style_name}_{model_mode}_epoch_total_loss.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    print(f"Epoch total loss plot saved to {loss_plot_path}")
    
    # Combined epoch losses
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, epoch_train_style_losses, 'o-', label='Train')
    plt.plot(epochs, epoch_test_style_losses, 's--', label='Test')
    plt.title('Style Loss')
    plt.grid(True, ls="--", alpha=0.7)
    plt.xticks(epochs)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, epoch_train_content_losses, 'o-', label='Train')
    plt.plot(epochs, epoch_test_content_losses, 's--', label='Test')
    plt.title('Content Loss')
    plt.grid(True, ls="--", alpha=0.7)
    plt.xticks(epochs)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, epoch_train_tv_losses, 'o-', label='Train')
    plt.plot(epochs, epoch_test_tv_losses, 's--', label='Test')
    plt.title('TV Loss')
    plt.grid(True, ls="--", alpha=0.7)
    plt.xticks(epochs)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs, epoch_train_total_losses, 'o-', label='Train')
    plt.plot(epochs, epoch_test_total_losses, 's--', label='Test')
    plt.title('Total Loss')
    plt.grid(True, ls="--", alpha=0.7)
    plt.xticks(epochs)
    plt.legend()
    
    plt.suptitle(f'Epoch-wise Loss Comparison - {style_name} - {model_mode} model', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    loss_plot_path = os.path.join(plots_dir, f'{style_name}_{model_mode}_epoch_combined_losses.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    print(f"Epoch combined loss plot saved to {loss_plot_path}")


def save_loss_data(train_style_losses, train_content_losses, train_tv_losses, train_total_losses,
                  test_style_losses, test_content_losses, test_tv_losses, test_total_losses,
                  epoch_train_style_losses, epoch_train_content_losses, epoch_train_tv_losses, epoch_train_total_losses,
                  epoch_test_style_losses, epoch_test_content_losses, epoch_test_tv_losses, epoch_test_total_losses,
                  plots_dir, style_name, model_mode):
    """Save loss data for later analysis or comparison"""
    data_dir = os.path.join(plots_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Save batch-wise losses
    batch_losses = {
        'train_style_losses': np.array(train_style_losses),
        'train_content_losses': np.array(train_content_losses),
        'train_tv_losses': np.array(train_tv_losses),
        'train_total_losses': np.array(train_total_losses),
        'test_style_losses': np.array(test_style_losses),
        'test_content_losses': np.array(test_content_losses),
        'test_tv_losses': np.array(test_tv_losses),
        'test_total_losses': np.array(test_total_losses)
    }
    
    batch_file = os.path.join(data_dir, f'{style_name}_{model_mode}_batch_losses.npz')
    np.savez(batch_file, **batch_losses)
    
    # Save epoch-wise losses
    epoch_losses = {
        'epoch_train_style_losses': np.array(epoch_train_style_losses),
        'epoch_train_content_losses': np.array(epoch_train_content_losses),
        'epoch_train_tv_losses': np.array(epoch_train_tv_losses),
        'epoch_train_total_losses': np.array(epoch_train_total_losses),
        'epoch_test_style_losses': np.array(epoch_test_style_losses),
        'epoch_test_content_losses': np.array(epoch_test_content_losses),
        'epoch_test_tv_losses': np.array(epoch_test_tv_losses),
        'epoch_test_total_losses': np.array(epoch_test_total_losses)
    }
    
    epoch_file = os.path.join(data_dir, f'{style_name}_{model_mode}_epoch_losses.npz')
    np.savez(epoch_file, **epoch_losses)
    
    print(f"Loss data saved to {data_dir}")


def main():
    parser = argparse.ArgumentParser(description='style transfer in pytorch')

    parser.add_argument("--model_mode", type=str, default="slim",
                        help="select the pattern of model, normal or slim")
    parser.add_argument("--dataset", type=str, default="./dataset", help="path to a training dataset")
    parser.add_argument("--test_dataset", type=str, default="./MAME-dataset", help="path to a testing dataset")
    parser.add_argument("--style_image", type=str, default="./style_imgs/Starry_Night.jpg",
                        help="path to a style image to train")
    parser.add_argument("--VGG_path", type=str, default=None, help="VGG save path")
    parser.add_argument("--image_height", type=int, default=256,
                        help="image's height, which will be fed into model")
    parser.add_argument("--image_width", type=int, default=256,
                        help="image's width, which will be fed into model")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="a hyperparameter which determines to what extent newly "
                             "acquired information overrides old information")
    parser.add_argument("--batch_size", type=int, default=14,
                        help="the number of training examples in one forward/backward pass")
    parser.add_argument("--epoch", type=int, default=5,
                        help="one cycle through the full training dataset")
    parser.add_argument("--style_weight", type=float, default=1e5,
                        help="hyperparameter, control style learning")
    parser.add_argument("--content_weight", type=float, default=1e0,
                        help="hyperparameter, bring into some correspondence with content image")
    parser.add_argument("--tv_weight", type=float, default=1e-7,
                        help="make result image smooth")
    parser.add_argument("--model_folder", type=str, default="style-transfer",
                        help="path to save model")

    args = parser.parse_args()

    trainer(args)


def print_args(args):
    print("\n" + "* " * 30)
    print()
    print("model mode: {}".format(args.model_mode))
    print("training dataset path: {}".format(args.dataset))
    print("testing dataset path: {}".format(args.test_dataset))
    print("style image path: {}".format(args.style_image))
    print("VGG path: {}".format(args.VGG_path))
    print("image size: ({}, {})".format(args.image_height, args.image_width))
    print("learning rate: {}".format(args.learning_rate))
    print("batch size: {}".format(args.batch_size))
    print("epoch: {}".format(args.epoch))
    print("style weight: {}".format(args.style_weight))
    print("content weight: {}".format(args.content_weight))
    print("tv weight: {}".format(args.tv_weight))
    print("model folder: {}".format(args.model_folder))
    print("\n" + "* " * 30 + "\n")

if __name__ == '__main__':
    main()
