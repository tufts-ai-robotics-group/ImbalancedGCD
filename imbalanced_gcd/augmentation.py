from torchvision import transforms


INTERP = transforms.InterpolationMode.BICUBIC
IMAGENET_SHAPE = (3, 224, 224)
IMAGE_SIZE = IMAGENET_SHAPE[1]


def dino_norm_trans():
    # DINO normalization for ImageNet pretraining
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def sim_gcd_train(image_size=IMAGE_SIZE):
    # SimGCD training transform
    # uses random ColorJitter, while original implementation effectively didn't use it
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=INTERP),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        dino_norm_trans(),
    ])


def sim_gcd_test(image_size=IMAGE_SIZE):
    # SimGCD testing transform
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=INTERP),
        transforms.CenterCrop(image_size),
        dino_norm_trans(),
    ])
