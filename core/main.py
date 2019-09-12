from dircadrn import DIRCADRN


config = {
    'data': '../../../dataset/LPBA40/samples/',
    'epoch': 10,
    'n_iters': 4000,
    'batch_size': 16,
    'patch_size': 31,
    'kernel_size': 5,
    'delta': 7,
    'num_data': 36208,
    'lr': 0.01,
    'momentum': 0.99,
    'beta': 0.999,
    'weight_decay': 0.1,
    'ckpt_path': '../ckpt/'
}

learner = DIRCADRN(config)
learner.train(debug=False)
