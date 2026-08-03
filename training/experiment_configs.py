experiment_configs =[
    {
        "name": "mlp_128_128_bs128",
        "hidden_sizes": (128, 128),
        "batch_size": 128,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
    },

    {
        "name": "mlp_256_128_bs128",
        "hidden_sizes": (256, 128),
        "batch_size": 128,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,

    },

    {
        "name": "mlp_128_128_bs64",
        "hidden_sizes": (128, 128),
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,  
    },

    {
            "name": "mlp_256_128_bs32_lr3e-4",
            "hidden_sizes": (256, 128),
            "batch_size": 32,
            "learning_rate": 3e-4,
            "weight_decay": 0.0,  
    },
]