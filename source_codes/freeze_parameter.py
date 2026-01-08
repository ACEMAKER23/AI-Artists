def freeze_model(pipe):
    for param in pipe.vae.parameters():
        param.requires_grad = False

    for param in pipe.text_encoder.parameters():
        param.requires_grad = False

    for param in pipe.unet.parameters():
        param.requires_grad = False