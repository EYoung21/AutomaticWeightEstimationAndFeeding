def resize_image(image, target_size):
    """Resize the input image to the target size."""
    from PIL import Image

    # Convert the image to a PIL Image object if it's not already
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Resize the image
    resized_image = image.resize(target_size, Image.ANTIALIAS)
    return resized_image


def normalize_image(image):
    """Normalize the input image to have pixel values between 0 and 1."""
    import numpy as np

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Normalize the image
    normalized_image = image_array / 255.0
    return normalized_image


def augment_image(image):
    """Apply random augmentations to the input image."""
    from torchvision import transforms

    # Define a series of augmentations
    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])

    # Apply the augmentations
    augmented_image = augmentation_transforms(image)
    return augmented_image