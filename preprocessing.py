import pandas as pd
from sklearn.model_selection import train_test_split

from config import ModelArguments, DataArguments

def make_dataset():
    verbatim_df = pd.read_pickle(DataArguments.VERBATIM_DATA_PATH)
    edited_df = pd.read_pickle(DataArguments.EDITED_DATA_PATH)

    edited_df['title_quote'] = edited_df['title_quote'].map(lambda x:x[0])
    edited_df['title_quote'] = edited_df['title_quote'].map(lambda x:x[0])

    # verbatim: 0, Edited: 1
    verbatim_df['ve'] = 0
    edited_df['ve'] = 1

    train_edited_df, test_edited_df = train_test_split(edited_df, test_size=0.2, random_state=ModelArguments.seed)
    valid_edited_df, test_edited_df = train_test_split(test_edited_df, test_size=0.5, random_state=ModelArguments.seed)

    train_verbatim_df, test_verbatim_df = train_test_split(edited_df, test_size=0.2, random_state=ModelArguments.seed)
    valid_verbatim_df, test_verbatim_df = train_test_split(test_verbatim_df, test_size=0.5, random_state=ModelArguments.seed)

    train_df = pd.concat([train_edited_df, train_verbatim_df])
    valid_df = pd.concat([valid_edited_df, valid_verbatim_df])
    test_df = pd.concat([test_edited_df, test_verbatim_df])

    return train_df, valid_df, test_df