import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import pandas as pd

# Set fixed random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CNN_DIR = os.path.join(BASE_DIR, 'cnn_dataset')

CLASSES = ['New_Sealed', 'Like_New', 'Minor_Wear', 'Damaged']
SPLITS = {
    'train': 420,       # 70% of 600
    'validation': 90,   # 15% of 600
    'test': 90          # 15% of 600
}

# Image dimensions
IMG_SIZE = (128, 128)

def create_base_product(draw, category, width, height, bg_color):
    """Draw a base product silhouette / package."""
    # Background
    draw.rectangle([0, 0, width, height], fill=bg_color)
    
    # Shadow
    shadow_box = [20, 24, width - 16, height - 20]
    draw.rounded_rectangle(shadow_box, radius=8, fill=(200, 205, 215))
    
    # Main package box / product body
    box = [16, 20, width - 20, height - 24]
    
    colors = [
        (45, 85, 125), (60, 120, 90), (140, 70, 70), (100, 80, 130),
        (180, 110, 40), (70, 70, 80), (30, 100, 110), (110, 50, 90)
    ]
    primary_color = random.choice(colors)
    
    draw.rounded_rectangle(box, radius=8, fill=primary_color)
    
    # Product display screen / label area
    inner_box = [box[0] + 10, box[1] + 12, box[2] - 10, box[3] - 25]
    draw.rectangle(inner_box, fill=(240, 243, 248))
    
    # Label stripes & barcode lines
    for i in range(inner_box[0] + 6, inner_box[2] - 6, 6):
        draw.line([(i, inner_box[3] - 10), (i, inner_box[3] - 3)], fill=(40, 40, 40), width=random.choice([1, 2]))
        
    # Brand logo placeholder
    draw.ellipse([inner_box[0] + 8, inner_box[1] + 8, inner_box[0] + 24, inner_box[1] + 24], fill=primary_color)
    draw.text((inner_box[0] + 30, inner_box[1] + 8), "HIEN-QA", fill=(50, 50, 60))
    
    return box, inner_box, primary_color

def apply_condition_features(image, condition, box, inner_box):
    """Apply visual cues matching the 4 condition classes."""
    draw = ImageDraw.Draw(image)
    w, h = image.size
    
    if condition == 'New_Sealed':
        # Factory Holographic Seal Tape across top & center
        seal_color = (230, 215, 120)  # Gold/Hologram
        draw.rectangle([box[0] + 15, box[1] - 4, box[2] - 15, box[1] + 6], fill=seal_color, outline=(180, 160, 60))
        draw.text((box[0] + 22, box[1] - 3), "[FACTORY SEALED]", fill=(80, 60, 10))
        
        # Subtle glossy shrinkwrap highlight line
        draw.line([(box[0] + 4, box[1] + 4), (box[2] - 20, box[3] - 10)], fill=(255, 255, 255, 120), width=2)
        
    elif condition == 'Like_New':
        # Open flap / broken seal tape (cleanly unsealed, zero damage on body)
        seal_color = (210, 210, 210)
        draw.rectangle([box[0] + 15, box[1] - 3, box[0] + 40, box[1] + 5], fill=seal_color)
        draw.rectangle([box[2] - 40, box[1] - 3, box[2] - 15, box[1] + 5], fill=seal_color)
        # Opened gap
        draw.line([(box[0] + 40, box[1] + 1), (box[2] - 40, box[1] + 1)], fill=(20, 20, 20), width=2)
        
    elif condition == 'Minor_Wear':
        # Broken seal + light cosmetic scratch marks / crease lines
        draw.line([(box[0] + 10, box[1] + 2), (box[2] - 10, box[1] + 2)], fill=(30, 30, 30), width=2)
        
        # 3 to 6 minor scratches / surface scuffs
        for _ in range(random.randint(3, 6)):
            sx = random.randint(inner_box[0], inner_box[2] - 15)
            sy = random.randint(inner_box[1], inner_box[3] - 10)
            ex = sx + random.randint(5, 18)
            ey = sy + random.randint(-4, 6)
            draw.line([(sx, sy), (ex, ey)], fill=(120, 125, 130), width=1)
            
        # Slightly worn corner
        draw.polygon([(box[0], box[1]), (box[0] + 8, box[1]), (box[0], box[1] + 8)], fill=(170, 175, 180))
        
    elif condition == 'Damaged':
        # Severe cracks, punctured holes, crushed box deformities
        # 1. Broken seal
        draw.line([(box[0] + 5, box[1] + 2), (box[2] - 5, box[1] + 2)], fill=(10, 10, 10), width=3)
        
        # 2. Jagged Crack lines across the surface
        crack_color = (30, 20, 20)
        for _ in range(random.randint(2, 4)):
            cx = random.randint(box[0] + 10, box[2] - 20)
            cy = random.randint(box[1] + 10, box[3] - 20)
            points = [(cx, cy)]
            for _ in range(random.randint(4, 7)):
                cx += random.randint(-15, 15)
                cy += random.randint(-10, 15)
                points.append((cx, cy))
            draw.line(points, fill=crack_color, width=random.choice([2, 3]))
            
        # 3. Crushed Corner / Tear Patch
        tear_x = random.choice([box[0] + 2, box[2] - 18])
        tear_y = random.choice([box[1] + 4, box[3] - 20])
        draw.polygon([
            (tear_x, tear_y), (tear_x + 16, tear_y + 4),
            (tear_x + 12, tear_y + 16), (tear_x - 2, tear_y + 14)
        ], fill=(40, 35, 35), outline=(20, 20, 20))
        
        # 4. Red Defect / Damage QA Marker
        draw.line([(box[0] + 10, box[1] + 10), (box[2] - 10, box[3] - 10)], fill=(220, 40, 40, 180), width=2)
        draw.line([(box[0] + 10, box[3] - 10), (box[2] - 10, box[1] + 10)], fill=(220, 40, 40, 180), width=2)

def generate_image(condition, img_id):
    """Generate a single PIL synthetic condition image."""
    bg_colors = [(245, 248, 252), (240, 242, 246), (235, 240, 245), (250, 250, 252)]
    bg = random.choice(bg_colors)
    
    img = Image.new('RGB', IMG_SIZE, color=bg)
    draw = ImageDraw.Draw(img)
    
    category = random.choice(['Electronics', 'Footwear', 'Apparel', 'Home', 'Toys'])
    box, inner_box, primary_col = create_base_product(draw, category, IMG_SIZE[0], IMG_SIZE[1], bg)
    
    apply_condition_features(img, condition, box, inner_box)
    
    # Slight subtle realistic noise
    if condition in ['Minor_Wear', 'Damaged']:
        # Optional subtle blur / grain
        if random.random() > 0.6:
            img = img.filter(ImageFilter.SMOOTH_MORE)
            
    return img

def build_dataset():
    print("=" * 80)
    print("GENERATING SYNTHETIC CNN RETURNED-PRODUCT CONDITION DATASET")
    print("=" * 80)
    
    metadata_rows = []
    
    for split_name, count_per_class in SPLITS.items():
        print(f"\nGenerating {split_name} split ({count_per_class} images x 4 classes = {count_per_class * 4} images)...")
        for cond in CLASSES:
            cond_dir = os.path.join(CNN_DIR, split_name, cond)
            os.makedirs(cond_dir, exist_ok=True)
            
            for idx in range(count_per_class):
                img_filename = f"{cond}_{split_name}_{idx:04d}.png"
                img_path = os.path.join(cond_dir, img_filename)
                rel_path = os.path.join(split_name, cond, img_filename)
                
                img = generate_image(cond, idx)
                img.save(img_path, "PNG")
                
                metadata_rows.append({
                    "image_id": f"IMG_{split_name.upper()[:3]}_{cond[:3]}_{idx:04d}",
                    "split": split_name,
                    "condition": cond,
                    "file_path": rel_path,
                    "width": IMG_SIZE[0],
                    "height": IMG_SIZE[1]
                })
                
    meta_df = pd.DataFrame(metadata_rows)
    meta_path = os.path.join(CNN_DIR, "metadata.csv")
    meta_df.to_csv(meta_path, index=False)
    
    print("\n" + "=" * 80)
    print(f"Dataset generated successfully! Total images: {len(meta_df):,}")
    print(f"Metadata saved to: {meta_path}")
    print("Class breakdown per split:")
    print(pd.crosstab(meta_df['split'], meta_df['condition']))
    print("=" * 80)

if __name__ == '__main__':
    build_dataset()
