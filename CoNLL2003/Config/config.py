config = {
    "CoNLL2003_Hyperparameters": {
        "heads": 6,
        "R": 5,
        "dim": 300,
        "max_len": 5000,
        'sparsegen': False,
        'sparsegen_lambda': 0.6,
        "in_dropout": 0.3,
        "attention_dropout": 0.3,
        "Transformer_dropout": 0.5,
        "layers": 3,
        "fc_dim": 600,
        "MLP_dropout": 0.3,
        "max_grad_norm": 1,
        "l2": 5e-5,
        "patience": 100,
        "Z_lambda": 2e-4,
        "lrate": 5e-4,
        "adaptive_span": True
    },

    "CoNLL2003_Sparse_Hyperparameters": {
        "heads": 6,
        "R": 5,
        "dim": 300,
        "max_len": 5000,
        'sparsegen': False,
        'sparsegen_lambda': 0.6,
        "attention_dropout": 0.3,
        "in_k": 0.3,
        "attention_dropout": 0.3,
        "Transformer_k": 0.3,
        "layers": 3,
        "fc_dim": 600,
        "MLP_dropout": 0.3,
        "max_grad_norm": 1,
        "l2": 5e-5,
        "patience": 100,
        "Z_lambda": 2e-4,
        "lrate": 5e-4,
        "adaptive_span": True
    },

    "CoNLL2003_Dynamic_Sparse_Hyperparameters": {
        "heads": 6,
        "R": 5,
        "dim": 300,
        "max_len": 5000,
        'sparsegen': False,
        'sparsegen_lambda': 0.6,
        "in_dropout": 0.3,
        "attention_dropout": 0.3,
        "Transformer_dropout": 0.5,
        "layers": 3,
        "fc_dim": 600,
        "MLP_dropout": 0.3,
        "max_grad_norm": 1,
        "l2": 5e-5,
        "patience": 100,
        "Z_lambda": 2e-4,
        "lrate": 5e-4,
        "adaptive_span": True,
        "sparse_lambda": 1e-8
    }
}