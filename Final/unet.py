import os
import random
import numpy as np
from glob import glob
from tqdm import tqdm
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from osgeo import gdal

# Enable GDAL exceptions to avoid noisy FutureWarning in GDAL 4.0+
gdal.UseExceptions()


# 实验为了可复现，设置 seed = 42
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


TRAIN_IMG_DIR = "D:/Projects/WilliamWei/train/image"
TRAIN_MSK_DIR = "D:/Projects/WilliamWei/train/label"
VAL_IMG_DIR = "D:/Projects/WilliamWei/test/image"
VAL_MSK_DIR = "D:/Projects/WilliamWei/test/label"

# 超参数
tile_size = 256  # 切片大小
overlap = 32  # 重叠（缓解边缘效应）
batch_size = 4
num_epochs = 30
lr = 1e-4
num_workers = max(2, min(8, os.cpu_count() or 0))
threshold = 0.5  # 二值化阈值（用于指标）

# 默认设备占位符
device = torch.device("cpu")

SAVE_DIR = "./model"
os.makedirs(SAVE_DIR, exist_ok=True)


def get_tif_files(folder_path):
    return sorted(
        [os.path.normpath(p) for p in glob(os.path.join(folder_path, "*.tif"))]
    )


class TiledRSDataset(Dataset):
    """
    以 (image_path, mask_path, xoff, yoff) 为索引，GDAL 按窗口读取；
    适配 Massachusetts Buildings 这类大图（常见 1500x1500）。
    """

    def __init__(
        self, image_paths, mask_paths, tile_size=256, overlap=32, augment=False
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.tile_size = tile_size
        self.overlap = overlap
        self.augment = augment
        self.index = self._build_index()

    def _build_index(self):
        idx = []
        step = self.tile_size - self.overlap
        for ip, mp in zip(self.image_paths, self.mask_paths):
            ds = gdal.Open(ip)
            w = ds.RasterXSize
            h = ds.RasterYSize
            # 右/下边界对齐：保证最后一个窗口覆盖到边缘
            xs = list(range(0, max(w - self.tile_size, 0) + 1, step))
            ys = list(range(0, max(h - self.tile_size, 0) + 1, step))
            if xs[-1] != w - self.tile_size:
                xs.append(w - self.tile_size)
            if ys[-1] != h - self.tile_size:
                ys.append(h - self.tile_size)

            for y in ys:
                for x in xs:
                    idx.append((ip, mp, x, y))
        return idx

    def __len__(self):
        return len(self.index)

    def _read_rgb_window(self, image_path, xoff, yoff, xsize, ysize):
        ds = gdal.Open(image_path)
        bands = []
        for i in range(1, 4):  # RGB 3 band
            b = ds.GetRasterBand(i).ReadAsArray(xoff, yoff, xsize, ysize)
            bands.append(b)
        arr = np.stack(bands, axis=0)  # [3,H,W]
        return arr

    def _read_mask_window(self, mask_path, xoff, yoff, xsize, ysize):
        ds = gdal.Open(mask_path)
        m = ds.GetRasterBand(1).ReadAsArray(xoff, yoff, xsize, ysize)  # [H,W]
        return m

    def _augment(self, x, y):
        # x: [3,H,W], y: [1,H,W]
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[2])
        if random.random() < 0.5:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[1])
        # 0/90/180/270 随机旋转
        k = random.randint(0, 3)
        if k > 0:
            x = torch.rot90(x, k, dims=[1, 2])
            y = torch.rot90(y, k, dims=[1, 2])
        return x, y

    def __getitem__(self, i):
        ip, mp, xoff, yoff = self.index[i]
        x = (
            self._read_rgb_window(
                ip, xoff, yoff, self.tile_size, self.tile_size
            ).astype(np.float32)
            / 255.0
        )
        y = (
            self._read_mask_window(
                mp, xoff, yoff, self.tile_size, self.tile_size
            ).astype(np.float32)
            / 255.0
        )

        x = torch.from_numpy(x)  # [3,H,W]
        y = torch.from_numpy(y).unsqueeze(0)  # [1,H,W]

        if self.augment:
            x, y = self._augment(x, y)

        # mask 强制二值（部分数据可能存在插值或非 0/255）
        y = (y > 0.5).float()
        return x, y


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """
    经典 U-Net 编码器-解码器 + 跳连结构。
    输出不做 sigmoid，配合 BCEWithLogitsLoss 更稳定。
    """

    def __init__(self, in_ch=3, out_ch=1, base=64):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)

        self.pool = nn.MaxPool2d(2)
        self.center = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.final = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        c = self.center(self.pool(e4))

        d4 = self.up4(c)
        d4 = self.dec4(torch.cat([e4, d4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([e3, d3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([e2, d2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([e1, d1], dim=1))

        logits = self.final(d1)
        return logits


def dice_loss_with_logits(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def compute_binary_metrics_from_logits(logits, targets, thr=0.5, eps=1e-7):
    """
    返回：pixel_acc, iou, dice, precision, recall, f1
    IoU/Dice 为语义分割常用指标。
    """
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()

    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    tn = ((1 - preds) * (1 - targets)).sum().item()

    pixel_acc = (tp + tn) / (tp + tn + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)
    return pixel_acc, iou, dice, precision, recall, f1


def read_rgb_full(image_path):
    ds = gdal.Open(image_path)
    arr = (
        np.stack(
            [ds.GetRasterBand(i).ReadAsArray() for i in range(1, 4)], axis=0
        ).astype(np.float32)
        / 255.0
    )
    return arr, ds


def save_geotiff_singleband(out_path, array_2d, ref_ds):
    driver = gdal.GetDriverByName("GTiff")
    h, w = array_2d.shape
    out = driver.Create(out_path, w, h, 1, gdal.GDT_Byte)
    out.SetGeoTransform(ref_ds.GetGeoTransform())
    out.SetProjection(ref_ds.GetProjection())
    out.GetRasterBand(1).WriteArray(array_2d)
    out.FlushCache()


@torch.no_grad()
def predict_full_image(model, image_path, tile_size=256, overlap=32, thr=0.5):

    model.eval()
    x_full, ds = read_rgb_full(image_path)  # [3,H,W]
    _, H, W = x_full.shape

    step = tile_size - overlap
    xs = list(range(0, max(W - tile_size, 0) + 1, step))
    ys = list(range(0, max(H - tile_size, 0) + 1, step))
    if xs[-1] != W - tile_size:
        xs.append(W - tile_size)
    if ys[-1] != H - tile_size:
        ys.append(H - tile_size)

    prob_acc = np.zeros((H, W), dtype=np.float32)
    cnt_acc = np.zeros((H, W), dtype=np.float32)

    for y0 in ys:
        for x0 in xs:
            patch = x_full[:, y0 : y0 + tile_size, x0 : x0 + tile_size]
            patch_t = torch.from_numpy(patch).unsqueeze(0).to(device)  # [1,3,t,t]
            logits = model(patch_t)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()  # [t,t]

            prob_acc[y0 : y0 + tile_size, x0 : x0 + tile_size] += prob
            cnt_acc[y0 : y0 + tile_size, x0 : x0 + tile_size] += 1.0

    prob_full = prob_acc / np.maximum(cnt_acc, 1e-6)
    pred_bin = (prob_full > thr).astype(np.uint8) * 255
    return pred_bin, ds, prob_full


def main():
    seed_everything(42)

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    log_path = os.path.join(SAVE_DIR, "output.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="a", encoding="utf-8")],
    )

    print("device =", device)
    logging.info("device = %s", device)

    train_imgs = get_tif_files(TRAIN_IMG_DIR)
    train_msks = get_tif_files(TRAIN_MSK_DIR)
    val_imgs = get_tif_files(VAL_IMG_DIR)
    val_msks = get_tif_files(VAL_MSK_DIR)

    print("train =", len(train_imgs), len(train_msks))
    print("val   =", len(val_imgs), len(val_msks))
    logging.info("train = %d %d", len(train_imgs), len(train_msks))
    logging.info("val   = %d %d", len(val_imgs), len(val_msks))
    assert len(train_imgs) == len(train_msks)
    assert len(val_imgs) == len(val_msks)

    persist_workers = num_workers > 0

    train_ds = TiledRSDataset(
        train_imgs, train_msks, tile_size=tile_size, overlap=overlap, augment=True
    )
    val_ds = TiledRSDataset(
        val_imgs, val_msks, tile_size=tile_size, overlap=overlap, augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persist_workers,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persist_workers,
        prefetch_factor=2,
    )

    print("train tiles =", len(train_ds))
    print("val tiles   =", len(val_ds))
    logging.info("train tiles = %d", len(train_ds))
    logging.info("val tiles   = %d", len(val_ds))

    model = UNet(in_ch=3, out_ch=1).to(device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print("params =", params_m, "M")
    logging.info("params = %.3f M", params_m)

    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_iou = -1.0
    best_path = os.path.join(SAVE_DIR, "best_unet.pth")

    for epoch in range(1, num_epochs + 1):
        model.train()
        tr_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [train]"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=scaler.is_enabled()
            ):
                logits = model(x)
                loss = bce(logits, y) + 0.5 * dice_loss_with_logits(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item()

        tr_loss /= max(len(train_loader), 1)

        model.eval()
        va_loss = 0.0
        agg = {"acc": 0, "iou": 0, "dice": 0, "prec": 0, "rec": 0, "f1": 0}
        n_batches = 0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [val]"):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.amp.autocast(
                    device_type=device.type, enabled=scaler.is_enabled()
                ):
                    logits = model(x)
                    loss = bce(logits, y) + 0.5 * dice_loss_with_logits(logits, y)
                va_loss += loss.item()

                acc, iou, dice, prec, rec, f1 = compute_binary_metrics_from_logits(
                    logits, y, thr=threshold
                )
                agg["acc"] += acc
                agg["iou"] += iou
                agg["dice"] += dice
                agg["prec"] += prec
                agg["rec"] += rec
                agg["f1"] += f1
                n_batches += 1

        va_loss /= max(len(val_loader), 1)
        for k in agg:
            agg[k] /= max(n_batches, 1)

        print(f"\nEpoch {epoch}: train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")
        print(
            "VAL metrics:",
            f"Acc={agg['acc']:.4f}",
            f"IoU={agg['iou']:.4f}",
            f"Dice={agg['dice']:.4f}",
            f"P={agg['prec']:.4f}",
            f"R={agg['rec']:.4f}",
            f"F1={agg['f1']:.4f}",
        )
        logging.info(
            "Epoch %d: train_loss=%.4f val_loss=%.4f Acc=%.4f IoU=%.4f Dice=%.4f P=%.4f R=%.4f F1=%.4f",
            epoch,
            tr_loss,
            va_loss,
            agg["acc"],
            agg["iou"],
            agg["dice"],
            agg["prec"],
            agg["rec"],
            agg["f1"],
        )

        if agg["iou"] > best_val_iou:
            best_val_iou = agg["iou"]
            torch.save(model.state_dict(), best_path)
            print("Saved best:", best_path)
            logging.info("Saved best: %s (IoU=%.4f)", best_path, best_val_iou)
        print("-" * 80)

    OUT_DIR = "./Output"
    os.makedirs(OUT_DIR, exist_ok=True)

    if val_imgs:
        model.load_state_dict(torch.load(best_path, map_location=device))
        test_image = val_imgs[0]
        pred_bin, ref_ds, prob_full = predict_full_image(
            model, test_image, tile_size=tile_size, overlap=overlap, thr=threshold
        )

        out_stem = os.path.splitext(os.path.basename(test_image))[0]
        out_path = os.path.join(OUT_DIR, f"{out_stem}_pred.tif")
        save_geotiff_singleband(out_path, pred_bin, ref_ds)
        print("saved:", out_path)
        logging.info("saved: %s", out_path)
    else:
        print("Skip inference: no validation images found.")
        logging.info("Skip inference: no validation images found.")


if __name__ == "__main__":
    main()
