import xml.etree.ElementTree as ET
import os
import random
import torch
import cv2
import numpy as np
import warnings
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Import custom processing modules
try:
    from noise import Synthetic_Noise_Application
    from color_scan_process import Color_Scan_Processing_A, Color_Scan_Processing_B
    from jpeg_process import JPEG_Artifact_Mitigation_C
except ImportError:
    print("WARNING: Custom modules not found. Image processing modes will be disabled.")
    def Synthetic_Noise_Application(img, **kwargs): return img
    def Color_Scan_Processing_A(img): return img 
    def Color_Scan_Processing_B(img): return img
    def JPEG_Artifact_Mitigation_C(img): return img

warnings.filterwarnings("ignore")

# ==========================================
# Configuration
# ==========================================
class Config:
    # Task: Style/Book Classification
    N_WAY = 3
    N_SHOT = 10

    # Environment
    BATCH_SIZE = 32        
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    MODEL_NAME = 'dinov2_vitb14'
    
    # Noise
    JPEG_QUALITY = 50      
    NOISE_LEVEL = 0.15     

    # Paths
    XML_DIR = '/Users/heartbeat/25-26-1/253/FS-Comic/Manga109_released_2023_12_07/annotations'
    IMG_DIR = '/Users/heartbeat/25-26-1/253/FS-Comic/Manga109_released_2023_12_07/images'
    
    # If VOLUME_LIST has items, it runs in "Tiny Mode" (only processing these volumes).
    # If VOLUME_LIST is None or empty, it runs in "Full Mode" (scanning all XMLs in XML_DIR).
    VOLUME_LIST = ['AisazuNihaIrarenai', 'SaladDays_vol01', 'EverydayOsakanaChan']
    # VOLUME_LIST = None

# ==========================================
# 1. Dataset Parsing & Metadata
# ==========================================
def parse_manga109_books(xml_root_dir, image_root_dir): 
    """
    Parses XML annotations to extract faces, labeling them by Book Title (Style).
    """
    if not os.path.exists(xml_root_dir):
        raise FileNotFoundError(f"Annotation directory not found: {xml_root_dir}")

    all_samples = []       
    
    # Mode Selection
    if Config.VOLUME_LIST and len(Config.VOLUME_LIST) > 0:
        print(f"Tiny Mode Active: Restricting to {len(Config.VOLUME_LIST)} volumes.")
        target_xmls = [f"{v}.xml" for v in Config.VOLUME_LIST]
    else:
        print("Full Mode Active: Scanning directory...")
        target_xmls = [f for f in os.listdir(xml_root_dir) if f.endswith('.xml')]

    valid_xmls = [f for f in target_xmls if os.path.exists(os.path.join(xml_root_dir, f))]
    print(f"Processing {len(valid_xmls)} volumes for style analysis...")

    for xml_file in valid_xmls:
        xml_path = os.path.join(xml_root_dir, xml_file)
        book_key = os.path.splitext(xml_file)[0] 
        current_img_dir = os.path.join(image_root_dir, book_key) 

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"Failed to parse {xml_file}: {e}")
            continue

        # Extract Faces
        pages_node = root.find('pages')
        if pages_node is not None:
            for page in pages_node.findall('page'):
                page_index = int(page.get('index'))
                img_path = os.path.join(current_img_dir, f"{page_index:03d}.jpg")
                
                if not os.path.exists(img_path): continue

                try:
                    with Image.open(img_path) as img:
                        img_width, img_height = img.size
                except:
                    continue
                
                for face in page.findall('face'):
                    # We take ALL faces in the book, regardless of character ID,
                    # because they all represent the author's style.
                    try:
                        xmin = int(face.get('xmin'))
                        ymin = int(face.get('ymin'))
                        xmax = int(face.get('xmax'))
                        ymax = int(face.get('ymax'))
                        
                        dx, dy = xmax - xmin, ymax - ymin
                        xmin = max(0, xmin - int(0.3 * dx))
                        xmax = min(img_width, xmax + int(0.3 * dx))
                        ymin = max(0, ymin - int(0.7 * dy))

                        if xmax > xmin and ymax > ymin:
                            all_samples.append({
                                'img_path': img_path,
                                'bbox': [xmin, ymin, xmax, ymax],
                                'label_book': book_key, # Label is Book Key (Style)
                            })
                    except ValueError:
                        continue 
                            
    print(f"Parsing complete. Extracted {len(all_samples)} style samples.")
    return all_samples

def build_style_meta(xml_dir, img_dir, min_shots):
    raw_samples = parse_manga109_books(xml_dir, img_dir)
    if not raw_samples: return []

    # Group by Book Key
    data_tree = defaultdict(list)
    for s in raw_samples:
        data_tree[s['label_book']].append(s)
    
    final_samples = []
    stats = {'classes': 0, 'images': 0} 
    
    for book_key, samples in data_tree.items():
        if len(samples) >= min_shots:
            final_samples.extend(samples)
            stats['classes'] += 1
            stats['images'] += len(samples)
            
    print(f"--- Metadata Built ---")
    print(f"Filtering: Min {min_shots} shots per book")
    print(f"Final: {stats['classes']} books (styles), {stats['images']} images")
    
    return final_samples

# ==========================================
# 2. Dataset Implementation
# ==========================================
class Manga109StyleDataset(Dataset):
    def __init__(self, meta_list, mode='clean', apply_noise=False): 
        self.meta_list = meta_list
        self.mode = mode
        self.apply_noise = apply_noise

        # Mapping based on Book Keys
        unique_books = sorted(list(set(s['label_book'] for s in meta_list)))
        self.cls2idx = {name: i for i, name in enumerate(unique_books)}
        self.idx2cls = {i: name for name, i in self.cls2idx.items()}

        self.transform_preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ])
        
        self.transform_postprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.meta_list)

    def __getitem__(self, idx):
        item = self.meta_list[idx]
        
        try:
            # 1. Load and Crop
            img = Image.open(item['img_path']).convert('RGB')
            img_pil = img.crop(item['bbox'])
            img_pil = self.transform_preprocess(img_pil) 

            # PIL RGB -> NumPy BGR
            img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # 2. Noise Injection
            if self.apply_noise:
                img_np = Synthetic_Noise_Application(
                    img_np, 
                    noise_level=Config.NOISE_LEVEL, 
                    jpeg_quality=Config.JPEG_QUALITY 
                )

            # 3. Restoration Strategy
            if self.mode == 'A':
                img_np = Color_Scan_Processing_A(img_np)
            elif self.mode == 'B':
                img_np = Color_Scan_Processing_B(img_np)
            elif self.mode == 'C':
                img_np = JPEG_Artifact_Mitigation_C(img_np)
            
            # NumPy BGR -> PIL RGB
            if img_np.ndim == 2:
                img_pil = Image.fromarray(img_np).convert('RGB')
            else:
                img_pil = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
                
            # 4. To Tensor
            img_tensor = self.transform_postprocess(img_pil)
            label = self.cls2idx[item['label_book']]
            
            return img_tensor, label
            
        except Exception as e:
            return torch.zeros((3, 224, 224)), 0

# ==========================================
# 3. Evaluation Logic (Few-Shot)
# ==========================================
def load_backbone():
    print(f"Loading Backbone: {Config.MODEL_NAME} ...")
    model = torch.hub.load('facebookresearch/dinov2', Config.MODEL_NAME)
    model.eval()
    model.to(Config.DEVICE)
    return model

def extract_features(model, dataloader):
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader, desc="Extracting Features"):
            imgs = imgs.to(Config.DEVICE)
            feats = F.normalize(model(imgs), p=2, dim=1)
            features.append(feats.cpu())
            labels.append(lbls)
    return torch.cat(features), torch.cat(labels)

def evaluate_protonet_episodes(full_s_feats, full_s_labels, full_q_feats, full_q_labels, 
                               n_way, k_shot, num_episodes=1000):
    print(f"\nStarting Evaluation: {n_way}-Way {k_shot}-Shot ({num_episodes} Episodes)...")
    
    # Map class index to features
    class_map = defaultdict(lambda: {'s': [], 'q': []})
    for f, l in zip(full_s_feats, full_s_labels): class_map[l.item()]['s'].append(f)
    for f, l in zip(full_q_feats, full_q_labels): class_map[l.item()]['q'].append(f)

    valid_classes = [c for c, d in class_map.items() if len(d['s']) >= k_shot and len(d['q']) > 0]
    
    if len(valid_classes) < n_way:
        print(f"Error: Not enough classes ({len(valid_classes)}) for {n_way}-Way classification.")
        return 0.0

    accs = []
    for _ in tqdm(range(num_episodes), desc="Episodes"):
        episode_cls = random.sample(valid_classes, n_way)
        s_feats, s_lbls, q_feats, q_lbls = [], [], [], []
        
        for i, cls_id in enumerate(episode_cls):
            # Support
            s_samples = random.sample(class_map[cls_id]['s'], k_shot)
            s_feats.extend(s_samples)
            s_lbls.extend([i] * k_shot)
            # Query
            q_samples = class_map[cls_id]['q']
            q_feats.extend(q_samples)
            q_lbls.extend([i] * len(q_samples))

        s_f = torch.stack(s_feats).to(Config.DEVICE)
        s_y = torch.tensor(s_lbls).to(Config.DEVICE)
        q_f = torch.stack(q_feats).to(Config.DEVICE)
        q_y = torch.tensor(q_lbls).to(Config.DEVICE)

        # ProtoNet
        prototypes = []
        for i in range(n_way):
            prototypes.append(s_f[s_y == i].mean(dim=0))
        prototypes = torch.stack(prototypes)

        dists = torch.cdist(q_f, prototypes)
        preds = torch.argmin(dists, dim=1)
        acc = (preds == q_y).float().mean().item()
        accs.append(acc)

    return np.mean(accs) * 100

def perform_dataset_split(meta_data, n_shot):
    # Group by Book Key for Style Classification
    grouped = defaultdict(list)
    for item in meta_data:
        grouped[item['label_book']].append(item)
        
    train_meta, test_meta = [], []
    for _, items in grouped.items():
        if len(items) <= n_shot: continue
        random.shuffle(items)
        train_meta.extend(items[:n_shot])
        test_meta.extend(items[n_shot:])
        
    return train_meta, test_meta

# ==========================================
# 4. Main Execution Flow
# ==========================================
if __name__ == "__main__":
    if os.path.exists(Config.XML_DIR) and os.path.exists(Config.IMG_DIR):
        # 1. Build Metadata
        meta_data = build_style_meta(Config.XML_DIR, Config.IMG_DIR, min_shots=Config.N_SHOT)
        
        if meta_data:
            train_meta, test_meta = perform_dataset_split(meta_data, Config.N_SHOT)
            
            # 2. Experiment Settings
            CURRENT_MODE = 'C'  
            APPLY_NOISE = True 
            
            print(f"\n--- Running Experiment: Style/Book Classification ---")
            print(f"Mode: {CURRENT_MODE} | Noise: {APPLY_NOISE} (Q={Config.JPEG_QUALITY})")

            # 3. Datasets
            support_ds = Manga109StyleDataset(train_meta, mode=CURRENT_MODE, apply_noise=APPLY_NOISE)
            query_ds   = Manga109StyleDataset(test_meta,  mode=CURRENT_MODE, apply_noise=APPLY_NOISE)
            
            s_loader = DataLoader(support_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
            q_loader = DataLoader(query_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
            
            # 4. Features
            model = load_backbone()
            print("Extracting Support Features...")
            s_feats, s_labels = extract_features(model, s_loader)
            print("Extracting Query Features...")
            q_feats, q_labels = extract_features(model, q_loader)
            
            # 5. Evaluate
            acc = evaluate_protonet_episodes(
                s_feats, s_labels, q_feats, q_labels,
                n_way=Config.N_WAY,
                k_shot=Config.N_SHOT,
                num_episodes=100
            )
            
            print("\n" + "="*50)
            print(f"Result: {Config.N_WAY}-Way {Config.N_SHOT}-Shot Style Classification")
            print(f"Mode: {CURRENT_MODE} | Noise: {APPLY_NOISE}")
            print(f"Accuracy: {acc:.2f}%")
            print("="*50)
            
        else:
            print("Dataset creation failed. Check paths.")
    else:
        print("Please configure Config.XML_DIR and Config.IMG_DIR correctly.")