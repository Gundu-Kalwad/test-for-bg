import os
import urllib.request
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from segment_anything import sam_model_registry, SamPredictor
from u2net import U2NET
from modnet import MODNet
from rvm.model import MattingNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Auto Download Models ------------------ #
def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"⬇️ Downloading {os.path.basename(dest)}...")
        urllib.request.urlretrieve(url, dest)
    else:
        print(f"✅ {os.path.basename(dest)} already exists.")

def download_models():
    os.makedirs("models", exist_ok=True)
    download_file("https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2net.pth", 
                  "models/u2net.pth")
    download_file("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", 
                  "models/sam_vit_h.pth")
    download_file("https://github.com/ZHKKKe/MODNet/releases/download/v0/modnet_photographic_portrait_matting.ckpt", 
                  "models/modnet_photographic_portrait_matting.ckpt")

# ------------------ Load Models ------------------ #
def load_models():
    print("📦 Loading models...")
    sam = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h.pth").to(device)
    sam_predictor = SamPredictor(sam)

    u2net = U2NET(3, 1)
    u2net.load_state_dict(torch.load("models/u2net.pth", map_location=device))
    u2net.to(device).eval()

    modnet = MODNet(backbone_pretrained=False).to(device)
    modnet.load_state_dict(torch.load("models/modnet_photographic_portrait_matting.ckpt", map_location=device))
    modnet.eval()

    rvm = MattingNetwork("mobilenetv3").to(device).eval()
    rvm.load_state_dict(torch.hub.load_state_dict_from_url(
        "https://huggingface.co/camenduru/RobustVideoMatting/resolve/main/rvm_mobilenetv3.pth"
    ))

    return sam_predictor, u2net, modnet, rvm

# ------------------ Helper Save ------------------ #
def save_rgba(fgr, alpha, path):
    rgba = np.dstack((fgr, alpha))
    rgba = (rgba * 255).astype(np.uint8)
    Image.fromarray(rgba).save(path)

# ------------------ Main Process ------------------ #
def combine_layers(image_path, output_path, sam_predictor, u2net, modnet, rvm):
    original = cv2.imread(image_path)
    image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    # SAM mask
    sam_predictor.set_image(image)
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])
    masks, _, _ = sam_predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=False)
    mask_sam = masks[0].astype(np.float32)

    # U2Net mask
    u2_input = transforms.Compose([transforms.Resize((320, 320)), transforms.ToTensor()])(
        Image.fromarray(image)).unsqueeze(0).to(device)
    with torch.no_grad():
        d1, *_ = u2net(u2_input)
        mask_u2 = d1[0][0].cpu().numpy()
        mask_u2 = cv2.resize(mask_u2, (w, h))
        mask_u2 = (mask_u2 - mask_u2.min()) / (mask_u2.max() - mask_u2.min())

    # MODNet mask
    mod_input = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])(
        Image.fromarray(image)).unsqueeze(0).to(device)
    with torch.no_grad():
        mask_mod = modnet(mod_input)[0][0].cpu().numpy()
        mask_mod = cv2.resize(mask_mod, (w, h))

    # Blend
    final_mask = (mask_sam * 0.4 + mask_u2 * 0.3 + mask_mod * 0.3)
    final_mask = np.clip(final_mask, 0, 1)

    # RVM refine
    image_tensor = torch.tensor(image / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    mask_tensor = torch.tensor(final_mask).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        fgr, pha, *_ = rvm(image_tensor, mask_tensor)
        fgr = fgr[0].permute(1, 2, 0).cpu().numpy()
        pha = pha[0, 0].cpu().numpy()[..., None]

    save_rgba(fgr, pha, output_path)
    print(f"✅ Saved: {output_path}")

# ------------------ Main Entry ------------------ #
def main():
    download_models()
    sam_predictor, u2net, modnet, rvm = load_models()

    input_dir = "input"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            in_path = os.path.join(input_dir, file)
            out_path = os.path.join(output_dir, f"final_{file.rsplit('.', 1)[0]}.png")
            combine_layers(in_path, out_path, sam_predictor, u2net, modnet, rvm)

if __name__ == "__main__":
    main()
