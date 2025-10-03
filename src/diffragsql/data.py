from datasets import load_dataset
def load_squad_splits(dataset_id='squad', split_train='train[:1%]', split_eval='validation[:1%]'):
    ds_train = load_dataset(dataset_id, split=split_train)
    ds_eval = load_dataset(dataset_id, split=split_eval)
    return ds_train, ds_eval
