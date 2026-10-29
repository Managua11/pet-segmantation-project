import segmentation_models_pytorch as smp


def build_model(encoder="resnet18", weights="imagenet"):
    return smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=1, activation=None)
