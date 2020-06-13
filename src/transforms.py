from torchvision import transforms


def get_transforms(train):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if train:
        trns = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        trns = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return trns
