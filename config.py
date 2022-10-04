from argparse import Namespace

ModelArguments = Namespace(
    seed=123,

    dimension_size = 768,
    hidden_size = 100,

    MAX_LEN = 512,
    learning_rate = 1e-6,
    weight_decay = 1e-7,
    batch_size = 4,
    num_workers=16,
    epochs = 500,
    iteration = 1043200,
    temperature=0.05,

    MODEL_DIR = './model/',
)

DataArguments = Namespace(
    EDITED_DATA_PATH ='data/edited_df.pkl',
    VERBATIM_DATA_PATH='data/verbatim_df.pkl',
    DATA_LOADER_SHUFFLE=True,
    DATA_LOADER_DROP_LAST=True,
)