import pandas as pd
from torch.utils.data import DataLoader
import warnings,transformers,logging,torch
from transformers import TrainingArguments,Trainer
from transformers import AutoModelForSequenceClassification,AutoTokenizer
import datasets
from datasets import load_dataset, Dataset, DatasetDict
# label encoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import torch.nn.functional as F
import wandb
import os

class Config:
    batch_size = 16
    max_epochs = 1
    lr = 1e-5
    warmup_ratio = 0.1
    weight_decay = 0.01
    max_len = 400
    fp16 = False
    gradient_accumulation_steps = 1
    model_name = 'microsoft/deberta-base'
    tokenizer_name = 'microsoft/deberta-base'
    log_dir = './logs'
    data_dir = 'data'
    train_file = 'train.csv'
    test_file = 'test.csv'
    wandb = True
    wandb_project = 'feedback_prize'
    logging_steps = 100
    run_name = 'deberta_without essaytext'
    output_dir = './output'
    overwrite_output_dir = True
    save_steps = 1000


def load_file(path,label_col):
    """
    load a file and return a dataframe with label column"""
    data = pd.read_csv(path)
    le = LabelEncoder()
    if label_col is not None and data[label_col].dtype == 'object':
        le.fit(data[label_col])
        data[label_col] = le.transform(data[label_col])
    data.rename(columns={label_col:'label'},inplace=True)
    return data

def split_data(train):
    essay_ids = train.essay_id.unique()
    val_prop = 0.2
    val_ids = essay_ids[:int(len(essay_ids)*val_prop)]
    val_idx = train[train.essay_id.isin(val_ids)].index
    train_idx = train[~train.essay_id.isin(val_ids)].index
    return train_idx,val_idx

def append_col_from_file(data,path,col_name,id_col):
    """
    append a column from a text files to a dataframe
    """
    data[col_name] = data[id_col].apply(lambda x: open(f'{path}{x}.txt').read())

def model_and_tokenizer(model_name,tokenizer_name,num_labels):
    """
    return a model and a tokenizer"""
    model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model,tokenizer

def concat_cols_using_sep(data,cols,sep,new_col,lower=True):
    """
    concat columns using sep
    params:
        data: dataframe
        cols: list of columns to concat
        sep: separator
        new_col: new column name
        lower: lower case the new column
    """
    if isinstance(cols,str):
        raise ValueError('cols must be a list')
    # if cols not in data.columns.tolist():
    #     raise ValueError('cols not in data')

    data[new_col] = data[cols[0]]
    for i in range(1,len(cols)):
        data[new_col] = data[new_col] + sep + data[cols[i]]
    if lower:
        data[new_col] = data[new_col].str.lower()
    return data

def config_wandb():
    # set enviroment variables
    os.environ['WANDB_API_KEY'] ='<your-wandb-api-key>'
    os.environ['WANDB_PROJECT'] = Config.wandb_project
    os.environ['WANDB_LOG_MODEL'] = 'True'

def tok_func(df):
    """
    tokenize a column using tokenizer
    params:
        df: dataframe
        cols_name: column name
        tokenizer: tokenizer
    """
    return tokenizer(df['inputs'],truncation=True,max_length=Config.max_len)

def get_dataset(df,tokenizer,train=True,to_remove=None,trn_idxs=None,val_idxs=None):
    ds = Dataset.from_pandas(df)
    to_remove = to_remove
    tok_ds = ds.map(tok_func, batched=True, remove_columns=to_remove)
    if train:
        return DatasetDict({'train':tok_ds.select(trn_idxs),'val':tok_ds.select(val_idxs)})
    else:
        return DatasetDict({'test':tok_ds})

def score(preds):
    return {'log loss': log_loss(preds.label_ids, F.softmax(torch.Tensor(preds.predictions)))}

def train_model(model,tokenizer,train_data,valid_data,config):
    """
    train a model
    params:
        model: model
        tokenizer: tokenizer
        train_data: training data
        valid_data: validation data
        config: config
    """
    
  
    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      train_dataset=train_data,
                      eval_dataset=valid_data,
            
                      args=TrainingArguments(
                          output_dir=config.output_dir,
                          overwrite_output_dir=config.overwrite_output_dir,
                          num_train_epochs=config.max_epochs,
                          per_device_train_batch_size=config.batch_size,
                          per_device_eval_batch_size=config.batch_size,
                         
                          weight_decay=config.weight_decay,
                          learning_rate=config.lr,
                          save_steps=config.save_steps,
                         
                          gradient_accumulation_steps=config.gradient_accumulation_steps,
                          fp16=config.fp16,
                          evaluation_strategy='steps',

                          report_to='wandb' if Config.wandb  else 'none',
                          logging_steps= config.logging_steps,
                         run_name=config.run_name,
                         load_best_model_at_end=True),compute_metrics=score)
    trainer.train()

if __name__ == '__main__':
    config_wandb()
    config = Config()
    train_data = load_file(os.path.join(config.data_dir,config.train_file),'discourse_effectiveness')

    model,tokenizer = model_and_tokenizer(config.model_name,config.tokenizer_name,num_labels=len(train_data['label'].unique()))

    train_data =concat_cols_using_sep(train_data,['discourse_type','discourse_text'],sep=tokenizer.sep_token,new_col='inputs')
    print(train_data['inputs'].head())
    train_idx,val_idx = split_data(train_data)
    dataset = get_dataset(train_data,tokenizer,train=True,to_remove=['discourse_text','discourse_type','inputs','discourse_id','essay_id'],trn_idxs=train_idx,val_idxs=val_idx)
    train_model(model,tokenizer,dataset['train'],dataset['val'],config)

 