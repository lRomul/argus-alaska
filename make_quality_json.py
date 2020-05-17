import json

from src.utils import get_image_quality
from src import config


if __name__ == "__main__":
    quality_dict = dict()
    count = 0

    for path in config.cover_dir.glob('*'):
        image_name = path.name
        quality = get_image_quality(config.cover_dir / image_name)
        quality_dict[image_name] = quality

        count += 1
        if count % 5000 == 0:
            print(count)

    with open(config.quality_json_path, 'w') as file:
        json.dump(quality_dict, file)

    print(f"Quality json saved to {config.quality_json_path}")
