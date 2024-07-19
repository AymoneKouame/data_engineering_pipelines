
import pandas as pd
import os
import subprocess
from google.cloud import bigquery
import nbformat as nbf
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor
import shutil
import time

client = bigquery.Client()

def client_read_gbq(query, dataset = os.getenv('WORKSPACE_CDR')):
    
    job_config = bigquery.QueryJobConfig(default_dataset=dataset)
    query_job = client.query(query, job_config =job_config)  # API request
    df = query_job.result().to_dataframe()

    return df

def read_from_bucket(filename, n = None, directory = 'notebooks/phenotype_data'):   
    my_bucket = os.getenv('WORKSPACE_BUCKET')
    ext = filename.split('.', -1)[-1]
    
    args = ["gsutil", "cp", f'{my_bucket}/{directory}/{filename}', './']
    output = subprocess.run(args, capture_output=True)
    print(output.stderr)    
    
    if ext == 'csv': df = pd.read_csv(filename, nrows = n)
    elif ext == 'xlsx': df = pd.read_excel(filename, nrows = n)
    elif ext =='tsv': df = pd.read_csv(filename, sep='\t', nrows = n)
    else: print('ERROR: file must be excel, tsv or csv.')
        
    return df

def save_to_bucket(df, filename, directory = 'notebooks/phenotype_data', save_index = False):
    my_bucket = os.getenv('WORKSPACE_BUCKET')
    ext = filename.split('.', -1)[-1]
    
    if ext == 'csv': df.to_csv(filename, index = save_index)
    elif ext == 'xlsx': df.to_csv(filename, index = save_index)
    elif ext =='tsv': df.to_csv(filename, sep='\t', index = save_index)
    else: print('ERROR: file must be excel, tsv or csv.')

    args = ["gsutil", "cp", f'{filename}', f'{my_bucket}/{directory}/']
    output = subprocess.run(args, capture_output=True)
    print(output.stderr)    
