# Py code to extract, clean, analyze and provide stats summary (including visualizations) of lab measurement data

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from datetime import datetime
from google.cloud import storage

import subprocess

from plotnine import *
import os
dataset = os.getenv('WORKSPACE_CDR')
my_bucket = os.getenv('WORKSPACE_BUCKET')

################################## load data that all notebooks use ##########################################
demographics_df = read_csv_from_bucket(name_of_file_in_bucket = 'demographics_table.csv')
ancestry_df = read_tsv('ancestry.tsv', col_types='ic-c-') %>% rename(person_id=research_id)
ancestry_df$ancestry_pred = toupper(ancestry_df$ancestry_pred)

#########################################################################

def load_to_bucket(source_filename, destination_blob_name = 'notebooks/all_x_all'):
    
    """Uploads a file to the bucket."""
    my_bucket = os.getenv('WORKSPACE_BUCKET')

    args = ["gsutil", "cp", "-R", f"./{source_filename}", f"{my_bucket}/{destination_blob_name}/{source_filename}"]
    output = subprocess.run(args, capture_output=True)
    print(output.stderr)
    print(f'\n file in bucket at {destination_blob_name}/{source_filename}')
    
    
def measurement_data(ancestor_cid):
    
    start = datetime.now() #started 6:16 pm ish
    
    #print(cid)
    df = pd.read_gbq(f"""
            SELECT DISTINCT person_id
                        , measurement_concept_id
                        , measurement_source_concept_id, LOWER(measurement_source_value) AS measurement_source_value
                        , value_as_number, range_low, range_high
                        , LOWER(unit_source_value) AS unit_source_value , operator_concept_id
                        , src_id
                        , measurement_datetime
                        ---, MAX(measurement_date) AS most_recent_measurement_date

            FROM `{dataset}.measurement` m 
            JOIN `{dataset}.measurement_ext` m_ext ON m.measurement_id = m_ext.measurement_id
            JOIN `{dataset}.concept_ancestor` ON descendant_concept_id = measurement_concept_id
            AND ancestor_concept_id IN ({ancestor_cid})
            WHERE m_ext.src_id LIKE '%EHR%' --GROUP BY 1,2,3,4,5,6,7,8,9 
            
            """)
    end = datetime.now()
    print(end - start)
    n_pids = df.person_id.nunique()
    
    print(f'N Pids :{n_pids}')
    
    return df

def map_concept_names(ancestor_cid, merge = 'yes'):

    df = pd.read_gbq(f"""
        SELECT DISTINCT measurement_concept_id, LOWER(STANDARD_CONCEPT_NAME) AS measurement_concept_name
        , measurement_source_concept_id, LOWER(SOURCE_CONCEPT_NAME) AS measurement_source_concept_name
        , unit_concept_id, LOWER(unit_concept_name) AS unit_concept_name
        , operator_concept_id, LOWER(operator_concept_name) AS operator_concept_name

        FROM `{dataset}.ds_measurement` me
        JOIN `{dataset}.concept_ancestor` ON descendant_concept_id = measurement_concept_id
        AND ancestor_concept_id IN ({ancestor_cid}) 
        """)
    return df

def percentile(n):
    def percentile_(x):
        return np.percentile(x[~np.isnan(x)], n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


def boxplot(measurement_df, c_name, fill_col, n_row= None, n_col = 1, w = 10, h = 12, facet_col = None):    
    #measurement_df['meas_and_unit'] = measurement_df['measurement_concept_name']+ ' ('+measurement_df['unit_concept_name']+')'

    plot = (ggplot(measurement_df, aes(x='src_id', y='value_as_number'#, fill=f'factor({fill_col})'
               , color =f'factor({fill_col})')) 
            + geom_boxplot()
            + labs(title = f"Comparing distributions of EHR {c_name} measures")
            + theme(axis_text_x = element_text(angle = 90, hjust = 1))
            + theme(figure_size=(w, h))
           )
    if facet_col is not None:     
        if n_row is None:
            n_row = measurement_df[facet_col].nunique()
        else:
            n_row = n_row
        plot = (plot
                + facet_wrap(facet_col, ncol = n_col, nrow = n_row, scales = 'free_y')
               )
             
    return plot


def check_df(measurement_df):
    cols = ['measurement_concept_name', 'measurement_source_concept_name', 'unit_concept_name'
        , 'operator_concept_name', 'value_as_number']

    qc_df1 = measurement_df[['person_id']+cols].drop_duplicates()
    qc_df1.loc[qc_df1.value_as_number.notna(), 'value_as_number'] = 'has value'
    qc_df1.loc[qc_df1.value_as_number.isna(), 'value_as_number'] = 'no value'
    qc_df1 = qc_df1.groupby(cols).nunique().sort_values('person_id')
    qc_df1.columns = ['n_pids']
    display(qc_df1)
    
    qc_df2 = measurement_df[['measurement_concept_name', 'value_as_number', 'unit_concept_name']].drop_duplicates()\
                .groupby(['measurement_concept_name', 'unit_concept_name'])\
                .agg({'value_as_number':['mean','median', 'min','max','std',percentile(1), percentile(25) , percentile(75), percentile(99)]})
    qc_df2.columns = [c[1] for c in qc_df2.columns]
    qc_df2 = qc_df2.reset_index()
    display(qc_df2)
    
    return [qc_df1, qc_df2]


def standard_cleaning(measurement_df): 
    print('''Standard Cleaning (for all measurements):\n 
- Drop rows with:
        - no value_as_number (can be done in SQL)
        - no units or 'no matching concept units' or units that do not make sense (e.g. 45666662)
 (This will most likely drop some EHR sites)\n 
- Harmonize units:
        - e.g. mg/dl = milligram per deciliter, Percent = percent''')
    
    harmonize_units_dd = {'mg/dl':'milligram per deciliter', 'Percent':'percent'
                          , 'meq/l':'milliequivalent per liter'}
    measurement_clean_df = measurement_df[(measurement_df.value_as_number.notna()) & (measurement_df.unit_concept_id.notna())]
    measurement_clean_df = measurement_clean_df[~(measurement_clean_df.unit_concept_name.isin(['no matching concept', 'no value']))]
    measurement_clean_df['unit_concept_name'] = measurement_clean_df['unit_concept_name'].replace(harmonize_units_dd)
    return measurement_clean_df

def drop_extreme_outliers(df, max_percentage_diff_threshold, max_percentile_threshold
                          , min_percentage_diff_threshold , min_percentile_threshold, c, u):
    
    print(f'''Definition: Drop extreme outliers, which is defined as rows where:
    - the max measurement value is {max_percentage_diff_threshold}% larger than the maximum of all values at or below the {max_percentile_threshold} percentile. 
    This threshold can be adjusted by using max_percentage_diff_threshold = n. The percentile threshold can also be adjusted using max_percentile_threshold = n.
    
    - and/or the min measurement value is {min_percentile_threshold}% smaller than the min of all values at or below the {min_percentile_threshold} percentile
    . This threshold can be adjusted by using min_percentage_diff_threshold = n. The percentile threshold can also be adjusted using min_percentile_threshold = n.\n\n\n''')
    
    # Drop extremely high values
    values = sorted(df.value_as_number.dropna())
    max_meas_value = max(values)
    #diff = max_meas_value - np.percentile(values, max_percentile_threshold)
    values_minus_max = [i for i in values if i < max_meas_value]
    max_values_2nd = max(values_minus_max)
    diff_perc1 = round(max_meas_value*100/max_values_2nd)
    
    if diff_perc1 >= max_percentage_diff_threshold:
        print(f'Dropping {max_meas_value} ({u})')
        df_clean = df[df.value_as_number < max_meas_value]
    else:
        df_clean = df.copy()
    
    
    # Drop extremely low values
    min_meas_value = min(values)
    #diff = min_meas_value - np.percentile(values, min_percentile_threshold)
    values_minus_min = [i for i in values if i > min_meas_value]
    min_values_2nd = min(values_minus_min)
    diff_perc2 = abs(round(min_meas_value*100/min_values_2nd))
    
    if diff_perc2 >= min_percentage_diff_threshold:
        print(f'Dropping {min_meas_value} ({u})')
        #print(f'For {c} in {u}, the min measurement value ({min_meas_value}) is {diff_perc2}% smaller than the min of all values at or below the 1 percentile ({min_values_2nd}).\n We will consider it an extreme outlier and drop it.')
        df_clean2 = df_clean[df_clean.value_as_number > min_meas_value]
    else:
        df_clean2 = df_clean.copy()
        
    return df_clean2

def drop_extreme_outliers_in_df(measurement_df, max_percentage_diff_threshold = 400, max_percentile_threshold = 99
                          , min_percentage_diff_threshold = 400, min_percentile_threshold = 1):
    measurement_clean2 = pd.DataFrame()
    
    print(f'''Definition: Drop extreme outliers, which is defined as rows where:
    - the max measurement value is {max_percentage_diff_threshold}% larger than the maximum of all values at or below the {max_percentile_threshold} percentile. 
    This threshold can be adjusted by using max_percentage_diff_threshold = n. The percentile threshold can also be adjusted using max_percentile_threshold = n.
    
    - and/or the min measurement value is {min_percentile_threshold}% smaller than the min of all values at or below the {min_percentile_threshold} percentile
    . This threshold can be adjusted by using min_percentage_diff_threshold = n. The percentile threshold can also be adjusted using min_percentile_threshold = n.\n\n\n''')


    for c in measurement_df.measurement_concept_name.unique():
        DF1 = measurement_df[(measurement_df.measurement_concept_name == c)]
        for u in DF1.unit_concept_name.unique():
            DF2 = DF1[DF1.unit_concept_name ==u]
            clean_DF2 = drop_extreme_outliers(DF2, max_percentage_diff_threshold = max_percentage_diff_threshold
                                              , max_percentile_threshold = max_percentile_threshold
                                              , min_percentage_diff_threshold = min_percentage_diff_threshold
                                              , min_percentile_threshold = min_percentile_threshold, c= c, u = u)
            measurement_clean2 = pd.concat([measurement_clean2, clean_DF2]) 

    return measurement_clean2

def final_data_output(clean_measurement_df):
    
    base_cols = ['person_id', 'value_as_number']
    df = clean_measurement_df[base_cols+['measurement_datetime']].drop_duplicates()

    df_latest= df.loc[df.groupby('person_id')['measurement_datetime'].idxmax()]
    df_latest = df_latest[base_cols].drop_duplicates()
    df_latest.columns = ['person_id','latest']
    df_latest = df_latest.reset_index(drop = True)

    df_units = clean_measurement_df[['person_id','unit_concept_name']].drop_duplicates()
    df_units['unit(s)'] = df_units.groupby('person_id')['unit_concept_name'].transform(lambda x: ', '.join(x.unique()))
    df_units = df_units.drop(['unit_concept_name'],1)
    df_units = df_units.drop_duplicates()
    
    df_stats = clean_measurement_df[base_cols]; df_stats[df_stats.value_as_number.notna()]    
    df_stats = df_stats.groupby([c for c in base_cols if c != 'value_as_number'])\
                                .agg({'value_as_number':['min','median','max','mean', 'count']})
    df_stats.columns = [c[1] for c in df_stats.columns]
    df_stats = df_stats.reset_index()

    df_final = df_stats.merge(df_units).merge(df_latest).drop_duplicates()
    return df_final,df_units