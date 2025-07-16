from google.cloud import bigquery
client = bigquery.Client()
import requests
import os
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
from IPython.core.interactiveshell import InteractiveShell

class utilities:    
    
    # Class to host all utility functions that are common to notebooks in this workspace
    def __init__(self):
        self.dataset = os.getenv('WORKSPACE_CDR')
        self.bucket = os.getenv('WORKSPACE_BUCKET') #bucket
        
    def client_read_gbq(self, query, jobconfig = None):
        # function to read data from BQ into py dataframe with the client
        if jobconfig is None: job_config = bigquery.QueryJobConfig(default_dataset=self.dataset)
        else: job_config = jobconfig
        query_job = client.query(query, job_config =job_config)  # API request
        df = query_job.result().to_dataframe()
        return df
        
    def get_federal_property_guideline_data(self, save = True):
        # function to get the federal poverty line data, wrangle it and use it to calculate UBR income
        # see https://aspe.hhs.gov/topics/poverty-economic-mobility/poverty-guidelines/poverty-guidelines-api

        import requests
        import pandas as pd
        from datetime import datetime

        # get data from the API
        print('Getting data from the aspe.hhs.gov API ...')
        start = datetime.now()
        poverty_https_link = "https://aspe.hhs.gov/topics/poverty-economic-mobility/poverty-guidelines/api/{YEAR}/{STATE_ABR}/{HOUSEHOLD_SIZE}"

        valid_years = list(range(1983, datetime.now().year+1)) #1983 to current year (2024)

        valid_states = ['us', 'hi', 'ak']
        valid_h_sizes = range(1,9)

        #######
        poverty_guideline_data = pd.DataFrame()
        for y in valid_years:
            for s in valid_states:
                for h in valid_h_sizes:
                    response = pd.DataFrame(requests.get(poverty_https_link.format(YEAR = y
                                                                , STATE_ABR = s, HOUSEHOLD_SIZE = h)).json()['data']
                                                                                        , index = ['']) 
                    poverty_guideline_data = pd.concat([poverty_guideline_data, response])


        ############## data wrangling########
        print('Data wrangling ...')
        fpl_guideline = poverty_guideline_data.rename(columns = {'income':'fpl'})
        for c in ['year','household_size','fpl']:
            fpl_guideline[c] = fpl_guideline[c].astype('int64')
        fpl_guideline['200_perc_fpl'] = fpl_guideline['fpl']*2

        ######
        print('  Adding data for household size >8')
        fpl_add_filename = 'poverty_guidelines-1983-2023.xlsx'
        args = ["gsutil", "cp", f"{self.bucket}/notebooks/{fpl_add_filename}", f"./"]
        output = subprocess.run(args, capture_output=True)
        print(output.stderr)
        
        keep_columns = ['Year', '8 Persons','$ For Each Additional Person (9+)']
        fpl_add_hi = pd.read_excel(fpl_add_filename, sheet_name=3, header =3)[keep_columns].loc[:40]; fpl_add_hi['state'] = 'hi'
        fpl_add_ak = pd.read_excel(fpl_add_filename, sheet_name=2, header =3)[keep_columns].loc[:40]; fpl_add_ak['state'] = 'ak'
        fpl_add_us = pd.read_excel(fpl_add_filename, sheet_name=1, header =3)[keep_columns].loc[:40]; fpl_add_us['state'] = 'us'

        fpl_add = pd.concat([fpl_add_hi, fpl_add_ak, fpl_add_us])
        fpl_add['9'] = fpl_add['8 Persons']+fpl_add[keep_columns[2]]

        ###
        fpl_add_clean = fpl_add.copy()
        for n in range(10,20):
            fpl_add_clean[f'{n}'] = fpl_add_clean[f'{n-1}']+fpl_add_clean[keep_columns[2]]

        fpl_add_clean = fpl_add_clean.drop(keep_columns[1:], axis = 1)
        fpl_add_clean = fpl_add_clean.melt(id_vars=['Year', 'state']).rename(columns = {'Year':"year", 'variable':'household_size', 'value':'fpl'})
        for c in ['year','household_size','fpl']:
            fpl_add_clean[c] = fpl_add_clean[c].astype('int64')

        fpl_add_clean['200_perc_fpl'] = fpl_add_clean['fpl']*2
        
        fpl_guideline = pd.concat([fpl_guideline, fpl_add_clean])

        ###################### save the file to the bucket
        if save == True:
            
            filename = 'poverty_guideline_data.csv'
            self.write_to_csv_excel(df_to_copy_final, filename = filename, keep_index = False)
 
        print('Done.')
        display(fpl_guideline.head(2))
        end = datetime.now(); print(f'Total processing time: {end-start}') #~5MIN

        return fpl_add_clean  

    def style_dataframe(self, df):
        print("-- 'style_dataframe()' is retired. Use 'p_display()' instead.")
        
    def p_display(self, df, format_counts = False, format_perc = False, fcount_cols= None, fperc_cols= None, title = ''):
        # Function to Style the dataframe - for pretty display only
        # Option to format integers as 10,000 and floats as percentage

        df_f = df.copy()

        if format_counts == True:
            if fcount_cols == None: fcount_cols = df_f.select_dtypes(include=np.integer).columns.tolist()
            elif fcount_cols != None: fcount_cols = fcount_cols 
            for c in fcount_cols:
                df_f[c] = ['{:,}'.format(i).replace('.0','') for i in df_f[c]] 

        if format_perc == True:
            if fperc_cols == None: fperc_cols = df_f.select_dtypes(include=float).columns.tolist()
            elif fperc_cols != None: fperc_cols = fperc_cols 
            for p in fperc_cols:
                df_f[p] = ['{:.1f}%'.format(i*100) for i in df_f[p]]


        s = df_f.style
        s.set_table_styles([
                {'selector': 'th.col_heading', 'props': 'text-align: center;'},
                {'selector': 'th.col_heading.level0', 'props': 'font-size: 1.2em;'},
                {'selector': 'td', 'props': 'text-align: right; font-weight: normal;font-size: 1.2em'},
                ], overwrite=False)

        for cols in df_f.columns:
            s.set_table_styles({
                cols: [{'selector': 'th', 'props': 'border-left: 0.1px solid #808080'},
                           {'selector': 'td', 'props': 'border-left: 0.1px solid #808080'}]
            }, overwrite=False, axis=0)

        print("\n\n",'\033[1m' + title.upper() + '\033[0m')

        display(s)
        

    def order_multiindex_df(self, df, order_keys):
        #function to arrange the multiindex columns of a dataframe based on the user defined order
        ordered_cols = []
        for k in order_keys: ordered_cols = ordered_cols+[c for c in df.columns if k in c]
        ordered_df = df[ordered_cols]
        ordered_df.columns = pd.MultiIndex.from_tuples(ordered_df.columns)
        
        return ordered_df
    
    def split_count_n_perc_col(self, final_df, col_to_split, col_dd = None, save = False, filename = None):
        # Function to split the values in the column into integer/num cols for easy copy/paste in excel
        # The if else Makes sure final_df has an index column. Line below ensure that    
        if type(final_df.index[0]) == int:
                print(f'Please set {final_df.columns[0]} as index before continuing. Aborting.')

        else:
            df_to_copy_final = pd.DataFrame()
            for orig_col in final_df.columns:
                df = final_df.copy()[[orig_col]]
                multiindex = [i for i in col_to_split if i in orig_col][0]
                if col_dd is None:count_col = 'Count'; perc_col = 'Perc'
                else: count_col = col_dd[multiindex]+' Count'; perc_col = col_dd[multiindex]+' Perc'
                    

                df[(f'{multiindex}', f'{count_col}')] = [int(i.split(' (')[0].replace(',','').replace('<','')) for i in df[orig_col]]
                df[(f'{multiindex}', f'{perc_col}')] = [float(i.split(' (')[1].replace('%)','')) for i in df[orig_col]]

                df_to_copy_final = pd.concat([df_to_copy_final, df], axis = 1).drop(orig_col, axis = 1)
                df_to_copy_final.columns = pd.MultiIndex.from_tuples(df_to_copy_final.columns)

            if save == True:
                self.write_to_csv_excel(df_to_copy_final, filename = filename)
            return df_to_copy_final
        
    def repl_less_than_20(self, df, cols = None):
            ## Function to replace counts that are less than 20 from a count column or a string column

        if cols == None:
            cols_to_transform = df.columns.tolist()
        else:
            cols_to_transform = cols  
        #####
        dd = dict()
        for c in cols_to_transform:
            df[c] = df[c].replace('-','0')
            if type(df[c][0]) == str:
                for i in df[c]: 
                    extr_i = str(i).replace(',','').replace('%','').split(' (')[0]
                    if int(extr_i) == 0: dd.update({i:'-'})
                    elif int(extr_i) <20: dd.update({i:'<20'})
                    else: dd.update({i:i})
                df[c] = df[c].replace(dd)            
            else:
                df.loc[df[c] <20, c] = '<20'

        return df 
    
    def write_to_csv_excel(self, df, filename, directory = 'notebooks/data', keep_index = True, save_to_bucket = True):
        #Function to save write any dataframe as excel or csv
        print(f"Default args = 'df, filename, directory = 'notebooks/data', keep_index = True, save_to_bucket = True'\n")
        if '.csv' in filename: df.to_csv(filename, index = keep_index)
        else: df.to_excel(filename, index = keep_index)

        if save_to_bucket == True:
            args = ["gsutil", "cp", f"./{filename}", f"{self.bucket}/{directory}"]
            output = subprocess.run(args, capture_output=True)
            print(output.stderr)