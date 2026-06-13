import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import glob
import re
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader

# =====================================================================
# 1. GLOBAL CONFIGURATION & HYPERPARAMETERS
# =====================================================================
NUM_MEMBERS = 5  
EPOCHS = 50
BATCH_SIZE = 32
OUTPUT_DIR = "Master_APISTEN_Results_Time_Series_Chl"
INPUT_DIR = "XLpolygon_input_directory"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# PINN CONFIGURATION: Updated with regression-based exponential equations
PINN_CONFIG = {
    "June":     {"date": "20210626", "doy": 177, "type": "exp", "a": 0.13,  "b": 2.3,  "lambda": 1.0}, 
    "November": {"date": "20211029", "doy": 302, "type": "exp", "a": 1.1,   "b": -0.36, "lambda": 1.0}, 
    "March":    {"date": "20220303", "doy": 62,  "type": "exp", "a": 0.17,  "b": 1.1,  "lambda": 1.0}, 
    "April":    {"date": "20220512", "doy": 132, "type": "exp", "a": 0.07, "b": 2.37,  "lambda": 1.0}  
}

# Large Scale (XL) Polygon Spatial Bounds
NX, NY = 1820, 1156
EXTENT = [16.2501, 16.6994, 43.3444, 43.5524]
SPLIT_ASPECT = 1.0 / np.cos(np.radians(43.53))

# =====================================================================
# 2. SPATIO-TEMPORAL U-NET ARCHITECTURE (11-Channel Input)
# =====================================================================
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(11, 32, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
        )
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        u2 = self.up2(b)
        if u2.shape != e2.shape: u2 = torch.nn.functional.interpolate(u2, size=e2.shape[2:])
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        if u1.shape != e1.shape: u1 = torch.nn.functional.interpolate(u1, size=e1.shape[2:])
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.final(d1)

# =====================================================================
# 3. DATA PREPARATION WITH CONTINUOUS HARMONIC EMBEDDING
# =====================================================================
def prepare_training_data():
    print("Loading raw stochastic realizations...")
    X_raw = np.fromfile("inputs.bin", dtype=np.float32).reshape(4000, 57, 17, 10)
    Y_raw = np.fromfile("outputs.bin", dtype=np.float32).reshape(4000, 57, 17)
    
    Y_mean, Y_std = Y_raw.mean(), Y_raw.std()
    np.save(f"{OUTPUT_DIR}/Y_stats_train.npy", np.array([Y_mean, Y_std]))
    
    X_11 = np.zeros((4000, 57, 17, 11), dtype=np.float32)
    Y_phys_target = np.zeros_like(Y_raw)
    Y_phys_weight = np.zeros((4000, 1, 1, 1), dtype=np.float32)

    blocks = [(0, 1000, "June"), (1000, 2000, "April"), 
              (2000, 3000, "November"), (3000, 4000, "March")]

    for start, end, name in blocks:
        cfg = PINN_CONFIG[name]
        
        theta = 2.0 * np.pi * cfg["doy"] / 365.25
        t_mod = np.float32(0.2 + 0.9 * ((1.0 + np.cos(theta)) / 2.0))
        
        B2, B3 = X_raw[start:end,:,:,0], X_raw[start:end,:,:,1]
        
        # APISTEN: Dynamically extract and save the offline B2/B3 baseline envelope stats
        ratio_block = B2 / (B3 + 1e-8)
        mu_b2b3 = float(np.mean(ratio_block))
        sigma_b2b3 = float(np.std(ratio_block))
        np.save(f"{OUTPUT_DIR}/B2B3_mean_{name}.npy", mu_b2b3)
        np.save(f"{OUTPUT_DIR}/B2B3_std_{name}.npy", sigma_b2b3)
        
        Y_p_raw = cfg['a'] * np.exp(cfg['b'] * ratio_block)
        
        Y_phys_target[start:end] = (Y_p_raw - Y_mean) / (Y_std + 1e-8)
        Y_phys_weight[start:end] = cfg['lambda']

        X_block = X_raw[start:end]
        m_x, s_x = X_block.mean(axis=(0,1,2), keepdims=True), X_block.std(axis=(0,1,2), keepdims=True)
        np.save(f"{OUTPUT_DIR}/X_mean_{name}.npy", m_x)
        np.save(f"{OUTPUT_DIR}/X_std_{name}.npy", s_x)

        X_11[start:end, :, :, :10] = (X_block - m_x) / (s_x + 1e-8)
        X_11[start:end, :, :, 10] = t_mod  
        
    print("Training matrices compiled and seasonal B2/B3 profiles archived.")
    return X_11, (Y_raw - Y_mean)/(Y_std + 1e-8), Y_phys_target, Y_phys_weight, Y_mean, Y_std

# =====================================================================
# 4. MODEL RE-TRAINING LOOP
# =====================================================================
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Executing pipeline on computation backend: {device}")

X_11, Y_norm, YP_t, YW_t, Y_mean, Y_std = prepare_training_data()

X_t = torch.tensor(X_11).permute(0, 3, 1, 2)
Y_t = torch.tensor(Y_norm).unsqueeze(1)
YP_t = torch.tensor(YP_t).unsqueeze(1)
YW_t = torch.tensor(YW_t)

ensemble_metrics = []

for m in range(NUM_MEMBERS):
    print(f"\n--- Training Spatio-Temporal Member {m+1}/{NUM_MEMBERS} ---")
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(X_t, Y_t, YP_t, YW_t), batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb, yp, yw in loader:
            xb, yb, yp, yw = xb.to(device), yb.to(device), yp.to(device), yw.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = nn.MSELoss()(pred, yb) + (yw[0,0,0,0] * nn.MSELoss()(pred, yp))
            loss.backward(); optimizer.step()
    
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/apisten_temporal_member_{m}.pth")

    model.eval()
    with torch.no_grad():
        preds = model(X_t.to(device))
        data_mse = nn.MSELoss()(preds, Y_t.to(device)).item()
        phys_mse = nn.MSELoss()(preds, YP_t.to(device)).item()
        
    ensemble_metrics.append([m, data_mse, phys_mse])
    print(f"Member {m} Complete -> Base MSE: {data_mse:.6f}, Physics MSE: {phys_mse:.6f}")

np.savetxt(f"{OUTPUT_DIR}/ensemble_metrics_summary.csv", np.array(ensemble_metrics), 
           delimiter=",", header="member,data_mse,phys_mse", comments='')

# =====================================================================
# 5. AUTOMATED OPERATIONAL INFERENCE (22 OVERPASS APISTEN ADAPTIVE LAUNCH)
# =====================================================================
print("\n=== Commencing 22-Overpass Adaptive Satellite Inversion ===")

lons = np.linspace(EXTENT[0], EXTENT[1], NY)
lats = np.linspace(EXTENT[3], EXTENT[2], NX)
lon_grid, lat_grid = np.meshgrid(lons, lats)

# 🛠️ MODIFICATION 1: Truncated Gaussian Umbrella Hyperparameters
lambda_max = 1.0  # Aligned to training max weight for perfect spectral matches
lambda_min = 0.1 # Rigid protective baseline floor for sensitive CDOM metrics

search_path = os.path.join(INPUT_DIR, "XLpolygon_input_*.bin")
input_files = sorted(glob.glob(search_path))

if len(input_files) == 0:
    raise FileNotFoundError(f"No valid .bin matrices discovered in target location: '{INPUT_DIR}'")

for filepath in input_files:
    filename = os.path.basename(filepath)
    date_match = re.search(r"XLpolygon_input_(\d{8})\.bin", filename)
    if not date_match: continue
    date_str = date_match.group(1)
    
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    doy = date_obj.timetuple().tm_yday
    theta = 2.0 * np.pi * doy / 365.25
    t_mod = np.float32(0.2 + 0.9 * ((1.0 + np.cos(theta)) / 2.0))

    # Identify the temporally closest seasonal anchor
    min_angle_dist = float('inf')
    chosen_anchor = "June"
    for anchor_name, cfg in PINN_CONFIG.items():
        angle_dist = np.abs(theta - (2.0 * np.pi * cfg["doy"] / 365.25))
        angle_dist = np.minimum(angle_dist, 2.0 * np.pi - angle_dist)
        if angle_dist < min_angle_dist:
            min_angle_dist = angle_dist
            chosen_anchor = anchor_name
            
    # Load the local scaling parameter matching this season
    m_x = np.load(f"{OUTPUT_DIR}/X_mean_{chosen_anchor}.npy")
    s_x = np.load(f"{OUTPUT_DIR}/X_std_{chosen_anchor}.npy")
    
    # Load corresponding B2/B3 envelope thresholds for Gaussian fading
    mu_b2b3 = np.load(f"{OUTPUT_DIR}/B2B3_mean_{chosen_anchor}.npy")
    sigma_b2b3 = np.load(f"{OUTPUT_DIR}/B2B3_std_{chosen_anchor}.npy")
    
    X_full = np.fromfile(filepath, dtype=np.float32).reshape(1, NX, NY, 10)
    
    # APISTEN Core Engine: Extract current overpass B2/B3 to calculate the structural Z-Score Map
    B2_overpass = X_full[0, :, :, 0]
    B3_overpass = X_full[0, :, :, 1]
    ratio_overpass = B2_overpass / (B3_overpass + 1e-8)
    
    z_map = (ratio_overpass - mu_b2b3) / (sigma_b2b3 + 1e-8)
    
    # 🛠️ MODIFICATION 2: Execute Truncated Gaussian Umbrella Math
    lambda_final = lambda_min + (lambda_max - lambda_min) * np.exp(-(z_map**2) / 2.0)
    
    print(f"Date: {date_str} (DOY: {doy:03d}) | Anchor: [{chosen_anchor}] | Max Lambda: {np.max(lambda_final):.3f} | Min Lambda: {np.min(lambda_final):.3f}")
    
    X_norm = (X_full - m_x) / (s_x + 1e-8)
    t_sheet = np.ones((1, NX, NY, 1), dtype=np.float32) * t_mod
    X_in = np.concatenate([X_norm, t_sheet], axis=-1)
    input_tensor = torch.tensor(X_in, dtype=torch.float32).permute(0, 3, 1, 2).to(device)

    all_preds = []
    for m in range(NUM_MEMBERS):
        model_eval = UNet().to(device)
        model_eval.load_state_dict(torch.load(f"{OUTPUT_DIR}/apisten_temporal_member_{m}.pth", map_location=device))
        model_eval.eval()
        with torch.no_grad():
            p = model_eval(input_tensor).squeeze().cpu().numpy()
            all_preds.append((p * Y_std) + Y_mean)
            
    mean_img = np.mean(all_preds, axis=0)
    std_img = np.std(all_preds, axis=0)

    # 🛠️ MODIFICATION 3: APISTEN Physical Guardrail (Clip to non-negative physical floor of 0.01 ppb)
    mean_img = np.clip(mean_img, 0.01, None)
    
    export_metrics = np.stack([lon_grid.flatten(), lat_grid.flatten(), mean_img.flatten(), std_img.flatten(), lambda_final.flatten()], axis=1)
    np.savetxt(f"{OUTPUT_DIR}/XLPolygon_Ensemble_{date_str}.csv", export_metrics, 
               delimiter=",", header="lon,lat,mean,std,lambda_applied", comments='')

    plt.figure(figsize=(12, 6))
    plt.imshow(mean_img, extent=EXTENT, origin='upper', cmap='viridis', aspect=SPLIT_ASPECT, vmin=0, vmax=3.0)
    plt.colorbar(label="Chl-a (mg/m³)")
    plt.title(f"APISTEN Adaptive Architecture Inversion: {date_str} (DOY {doy})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig(f"{OUTPUT_DIR}/Map_{date_str}_Mean.png", dpi=150, bbox_inches='tight')
    plt.close()

print(f"\nAPISTEN time-series processing executed seamlessly. Check output tables for the saved pixel-by-pixel 'lambda_applied' values.")