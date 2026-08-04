experiment_configs = [
    # ------------------------------------------------------------------
    # Current winner
    # ------------------------------------------------------------------
    {
        "name": "mlp_256_128_bs32_lr3e-4_do10",
        "hidden_sizes": (256, 128),
        "batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "dropout": 0.10,
    },

    # ------------------------------------------------------------------
    # Batch-size sweep
    # Everything else matches the winner.
    # ------------------------------------------------------------------
    {
        "name": "mlp_256_128_bs16_lr3e-4_do10",
        "hidden_sizes": (256, 128),
        "batch_size": 16,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "dropout": 0.10,
    },
    {
        "name": "mlp_256_128_bs64_lr3e-4_do10",
        "hidden_sizes": (256, 128),
        "batch_size": 64,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "dropout": 0.10,
    },
    {
        "name": "mlp_256_128_bs128_lr3e-4_do10",
        "hidden_sizes": (256, 128),
        "batch_size": 128,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "dropout": 0.10,
    },

    # ------------------------------------------------------------------
    # Dropout sweep
    # ------------------------------------------------------------------
    {
        "name": "mlp_256_128_bs32_lr3e-4_do00",
        "hidden_sizes": (256, 128),
        "batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "dropout": 0.00,
    },
    {
        "name": "mlp_256_128_bs32_lr3e-4_do05",
        "hidden_sizes": (256, 128),
        "batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "dropout": 0.05,
    },
    {
        "name": "mlp_256_128_bs32_lr3e-4_do15",
        "hidden_sizes": (256, 128),
        "batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "dropout": 0.15,
    },
    {
        "name": "mlp_256_128_bs32_lr3e-4_do20",
        "hidden_sizes": (256, 128),
        "batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "dropout": 0.20,
    },

    # ------------------------------------------------------------------
    # Learning-rate sweep
    # 1e-4 is conservative; 1e-3 and 3e-3 are increasingly aggressive.
    # ------------------------------------------------------------------
    {
        "name": "mlp_256_128_bs32_lr1e-4_do10",
        "hidden_sizes": (256, 128),
        "batch_size": 32,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "dropout": 0.10,
    },
    {
        "name": "mlp_256_128_bs32_lr1e-3_do10",
        "hidden_sizes": (256, 128),
        "batch_size": 32,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "dropout": 0.10,
    },
    {
        "name": "mlp_256_128_bs32_lr3e-3_do10",
        "hidden_sizes": (256, 128),
        "batch_size": 32,
        "learning_rate": 3e-3,
        "weight_decay": 0.0,
        "dropout": 0.10,
    },

    # ------------------------------------------------------------------
    # Capacity sweep
    # ------------------------------------------------------------------
    {
        "name": "mlp_64_64_bs32_lr3e-4_do10",
        "hidden_sizes": (64, 64),
        "batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "dropout": 0.10,
    },
    {
        "name": "mlp_128_64_bs32_lr3e-4_do10",
        "hidden_sizes": (128, 64),
        "batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "dropout": 0.10,
    },
    {
        "name": "mlp_256_256_bs32_lr3e-4_do10",
        "hidden_sizes": (256, 256),
        "batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "dropout": 0.10,
    },
    {
        "name": "mlp_512_256_bs32_lr3e-4_do10",
        "hidden_sizes": (512, 256),
        "batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "dropout": 0.10,
    },
]