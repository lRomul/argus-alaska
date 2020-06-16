import random
import numpy as np
import pandas as pd


from src import config


if __name__ == '__main__':
    random_state = 42
    val_samples = 5000

    random.seed(random_state)
    np.random.seed(random_state)

    cover_image_paths = sorted(config.cover_dir.glob("*"))
    image_names = [image_path.name for image_path in cover_image_paths]

    folds = [0] * val_samples + [1] * (len(image_names) - val_samples)
    random.shuffle(folds)

    train_df = pd.DataFrame({
        'name': image_names,
        'fold': folds
    })

    train_df.to_csv(config.train_folds_path, index=False)
    print(f"Train folds saved to '{config.train_folds_path}'")
