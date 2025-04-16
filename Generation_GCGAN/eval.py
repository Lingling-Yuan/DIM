import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import utils as vutils
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import argparse
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image

from models import Generator
from pytorch_fid import fid_score


def ssim_pytorch(img1, img2, window_size=11, size_average=True):
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                                for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def resize(img, size=256):
    return F.interpolate(img, size=size)


def batch_generate(zs, netG, batch=8):
    g_images = []
    with torch.no_grad():
        for i in range(len(zs) // batch):
            g_images.append(netG(zs[i * batch:(i + 1) * batch]).cpu())
        if len(zs) % batch > 0:
            g_images.append(netG(zs[-(len(zs) % batch):]).cpu())
    return torch.cat(g_images)


def batch_save(images, folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    for i, image in enumerate(images):
        vutils.save_image(image.add(1).mul(0.5), os.path.join(folder_name, f'{i}.jpg'))


def calculate_fid(fake_images_path, real_images_path, batch_size=20):
    with torch.no_grad():
        fid_value = fid_score.calculate_fid_given_paths([fake_images_path, real_images_path],
                                                          batch_size, device, 2048)
    return fid_value


def calculate_inception_score(images, batch_size=16, splits=1):
    weights = Inception_V3_Weights.DEFAULT
    inception_model = inception_v3(weights=weights, transform_input=False).to(device)
    inception_model.eval()
    preprocess = transforms.Compose([
        transforms.Lambda(lambda x: (x + 1) / 2),
        transforms.Resize((299, 299), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    preds = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch = torch.stack([preprocess(img) for img in batch]).to(device)
            pred = inception_model(batch).cpu().numpy()
            preds.append(pred)
    preds = np.concatenate(preds, axis=0)
    # 对 logits 计算 softmax
    preds = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
    scores = []
    for i in range(splits):
        part = preds[i * (len(preds) // splits):(i + 1) * (len(preds) // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, axis=0), 0)))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir)
                          if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def calculate_psnr_ssim(fake_images, real_images_path):
    real_images = CustomImageDataset(real_images_path, transform=transforms.ToTensor())
    real_loader = DataLoader(real_images, batch_size=1, shuffle=True)

    psnr_vals, ssim_vals = [], []
    with torch.no_grad():
        for i, real_img in enumerate(real_loader):
            real_img = real_img.to(device)
            fake_img = fake_images[i % len(fake_images)]

            fake_img = ((fake_img + 1) / 2).unsqueeze(0).to(device)
            real_img_resized = resize(real_img, fake_img.shape[2:])
            psnr_val = psnr(fake_img.cpu().numpy().transpose(0, 2, 3, 1),
                            real_img_resized.cpu().numpy().transpose(0, 2, 3, 1), data_range=1)
            ssim_val = ssim_pytorch(fake_img, real_img_resized)
            psnr_vals.append(psnr_val)
            ssim_vals.append(ssim_val.item())
            if i >= len(fake_images) - 1:
                break

    return np.mean(psnr_vals), np.mean(ssim_vals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate images and evaluate metrics.')
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--artifacts', type=str, default=".", help='Path to artifacts.')
    parser.add_argument('--cuda', type=int, default=0, help='Index of GPU to use')
    parser.add_argument('--start_iter', type=int, default=4)
    parser.add_argument('--end_iter', type=int, default=8)
    parser.add_argument('--dist', type=str, default='.')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch', default=16, type=int, help='Batch size')
    parser.add_argument('--n_sample', type=int, default=50)
    parser.add_argument('--big', action='store_true')
    parser.add_argument('--im_size', type=int, default=256)
    parser.add_argument('--multiplier', type=int, default=10000, help='Multiplier for model number')
    parser.set_defaults(big=False)
    args = parser.parse_args()

    noise_dim = 256
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    net_ig = Generator(ngf=64, nz=noise_dim, nc=3, im_size=args.im_size)
    net_ig.to(device)

    for epoch in [args.multiplier * i for i in range(args.start_iter, args.end_iter + 1)]:
        ckpt = os.path.join(args.artifacts, "models", f"{epoch}.pth")
        checkpoint = torch.load(ckpt, map_location=lambda a, b: a)
        checkpoint['g'] = {k.replace('module.', ''): v for k, v in checkpoint['g'].items()}
        net_ig.load_state_dict(checkpoint['g'])

        print(f'Load checkpoint success, epoch {epoch}')
        net_ig.to(device)

        del checkpoint

        dist = os.path.join(f'eval_{epoch}', 'img')
        os.makedirs(dist, exist_ok=True)

        fake_images = []
        with torch.no_grad():
            for i in tqdm(range(args.n_sample // args.batch)):
                noise = torch.randn(args.batch, noise_dim).to(device)

                g_imgs = net_ig(noise)[0]
                g_imgs = resize(g_imgs, args.im_size)
                fake_images.extend(g_imgs.cpu())
                for j, g_img in enumerate(g_imgs):

                    vutils.save_image(g_img.add(1).mul(0.5),
                                      os.path.join(dist, f'{i * args.batch + j}.png'))

        fake_images_path = os.path.join(dist, 'generated_images')
        batch_save(fake_images, fake_images_path)

        real_images_path = ".../data/CRC/Normal"
        fid_value = calculate_fid(fake_images_path, real_images_path)
        print(f'FID score for epoch {epoch}: {fid_value}')

        inception_score, inception_score_std = calculate_inception_score(fake_images)
        print(f'Inception score for epoch {epoch}: {inception_score} ± {inception_score_std}')

        psnr_val, ssim_val = calculate_psnr_ssim(fake_images, real_images_path)
        print(f'PSNR for epoch {epoch}: {psnr_val}')
        print(f'SSIM for epoch {epoch}: {ssim_val}')
