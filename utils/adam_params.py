def custom_params(model, weight_decay=0, lr=1e-3, lr_transfo=3e-5, lr_decay=1):
    """
    Custom parameters for Bert Models to handle weight decay and differentiated learning rates.

    Arguments:
        model {torch model} -- Model to get parameters on

    Keyword Arguments:
        weight_decay {float} -- Weight decay (default: {0})
        lr {float} -- Learning rate of layers not belongign to the transformer, i.e. not pretrained (default: {1e-3})
        lr_transfo {float} -- Learning rate of the layer the of the transformer the closer to the output (default: {3e-5})
        lr_decay {float} -- How much to multiply the lr_transfo by when going deeper in the model (default: {1})

    Returns:
        torch opt_params -- Parameters to feed the optimizer for the model
    """
    opt_params = []
    no_decay = ["bias", "LayerNorm.weight"]
    nb_blocks = len(model.transformer.encoder.layer)

    for n, p in model.named_parameters():
        wd = 0 if any(nd in n for nd in no_decay) else weight_decay

        if "transformer" in n and "pooler" not in n:
            lr_ = lr_transfo
            if "transformer.embeddings" in n:
                lr_ = lr_transfo * lr_decay ** (nb_blocks)
            else:
                for i in range(nb_blocks):  # for bert base
                    if f"layer.{i}." in n:
                        lr_ = lr_transfo * lr_decay ** (nb_blocks - 1 - i)
                        break
        else:
            lr_ = lr

        opt_params.append({
            "params": [p],
            "weight_decay": wd,
            'lr': lr_,
        })
        # print(n, lr_, wd)
    return opt_params
