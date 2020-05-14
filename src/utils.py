

def initialize_amp(model,
                   opt_level='O1',
                   keep_batchnorm_fp32=None,
                   loss_scale='dynamic'):
    from apex import amp
    model.nn_module, model.optimizer = amp.initialize(
        model.nn_module, model.optimizer,
        opt_level=opt_level,
        keep_batchnorm_fp32=keep_batchnorm_fp32,
        loss_scale=loss_scale
    )
    model.amp = amp
