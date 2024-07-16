def get_encoder(encoding, input_dim=3, multires=6, degree=4):
    if encoding == "None":
        return lambda x, **kwargs: x, input_dim

    elif encoding == "frequency":
        from src.models.nerf.encoders.freqencoder import FreqEncoder
        encoder = FreqEncoder(input_dim=input_dim, degree=multires)

    elif encoding == "sphere_harmonics":
        from src.models.nerf.encoders.shencoder import SHEncoder
        encoder = SHEncoder(input_dim=input_dim, degree=degree)

    else:
        raise NotImplementedError("Unknown encoding mode, choose from [None, frequency, sphere_harmonics]")

    return encoder, encoder.output_dim
