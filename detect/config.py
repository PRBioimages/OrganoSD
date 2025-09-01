organ512 = {
    'cfg_unet': [16, 32, 64, 128, 256, 512, 1024, 2048],
    'num_classes': 1,
    'lr_steps': (800, 2400, 4800, 6000),
    'max_iter': 120000,
    'feature_maps': [32, 16, 8, 4],
    'min_dim': 512,
    'steps': [16, 32, 64, 128],
    'min_sizes': [12, 28, 55, 120],
    'max_sizes': [35, 50, 88, 150],
    'aspect_ratios': [[], [2], [2, 3], [2, 3]],
    'prior_num': [2, 4, 6, 6],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

organ = {
    'num_classes': 1,
    'lr_steps': (800, 2400, 4800, 6000),
    'max_iter': 50000,
    'feature_maps': [38, 19, 10, 5, 3, 2],
    'min_dim': 732,
    'steps': [19, 38, 73, 64, 244, 732],
    'min_sizes': [42, 73, 146, 274, 402, 545],
    'max_sizes': [73, 146, 274, 402, 545, 659],
    'aspect_ratios': [1, 1, [2], [2], [2,3], 1],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'ogan',
}


organ5 = {
    'num_classes': 1,
    'lr_steps': (800, 2400, 4800, 6000),
    'max_iter': 50000,
    'feature_maps': [32, 16, 8, 4, 2],
    'min_dim': 512,
    # 'map_layers_sim': [512, 256, 128, 64, 32],
    # 'map_layers_com': [1024, 512, 256, 128, 64],
    'steps': [16, 32, 64, 128, 256],
    'min_sizes': [30, 54, 78, 102, 136],
    'max_sizes': [59, 81, 103, 150, 160],
    'prior_num': [2, 6, 8, 8, 8],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'ogan',
}


