import os
import pandas as pd
from PIL import Image
import numpy as np

def load_and_save_data(data_dir="data/pizza_not_pizza", image_size=(64, 64)):
    data = []
    for label in ["pizza", "not_pizza"]:
        folder = os.path.join(data_dir, label)
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                data.append({
                    "filepath": os.path.join(folder, fname),
                    "label": label
                })
    df = pd.DataFrame(data)

    X = np.array([np.array(Image.open(p).convert("RGB").resize(image_size)) for p in df['filepath']])
    y = np.array([1 if lbl=="pizza" else 0 for lbl in df['label']])

    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/X.npy", X)
    np.save("data/processed/y.npy", y)
    return X, y
