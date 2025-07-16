# Code to run notebook in the background

import glob
import gzip
import nbformat
from nbconvert.preprocessors import CellExecutionError
from nbconvert.preprocessors import ExecutePreprocessor
import os
import shutil
import tensorflow as tf
import time
import subprocess

summary_notebooks_to_run = [f for f in os.listdir() if f.endswith('_summary.ipynb')]

def get_kernel(kernel):
    return 'ir' if kernel.lower() == 'r' else 'python3'

def run_notebook(NOTEBOOK_TO_RUN, KERNEL = 'R'):
    
    KERNEL_NAME = get_kernel(KERNEL)    
    OUTPUT_NOTEBOOK = NOTEBOOK_TO_RUN
    
    with open(NOTEBOOK_TO_RUN) as f_in:
        nb = nbformat.read(f_in, as_version=4)
        ep = ExecutePreprocessor(timeout=-1, kernel_name=KERNEL_NAME)
        try:
            out = ep.preprocess(nb, {'metadata': {'path': ''}})
        except CellExecutionError:
            out = None
            print(f'''Error executing the notebook "{NOTEBOOK_TO_RUN}".
            See notebook "{OUTPUT_NOTEBOOK}" for the traceback.''')
        finally:
            with open(OUTPUT_NOTEBOOK, mode='w', encoding='utf-8') as f_out:
                nbformat.write(nb, f_out)
            # Save the executed notebook to the workspace bucket.
            output_notebook_path = os.path.join(os.getenv('WORKSPACE_BUCKET'), 'notebooks', OUTPUT_NOTEBOOK)
            tf.io.gfile.copy(src=OUTPUT_NOTEBOOK, dst=output_notebook_path,overwrite=True)
            print(f'Wrote executed notebook to {output_notebook_path}.')
            

# Usage
for NOTEBOOK_TO_RUN in summary_notebooks_to_run:
    run_notebook(NOTEBOOK_TO_RUN, KERNEL = 'R')