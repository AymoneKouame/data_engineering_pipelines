from google.cloud import bigquery
client = bigquery.Client()
import os
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
from utilities import utilities as u

class genomics_by_phenotypes:    
    ## Class to get ubr flags for each person in the current dataset
    
    def __init__(self):
        self.dataset = os.getenv('WORKSPACE_CDR')#dataset
        self.bucket = os.getenv('WORKSPACE_BUCKET') #bucket

    def client_read_gbq(self, query, jobconfig = None):
    
        if jobconfig is None: job_config = bigquery.QueryJobConfig(default_dataset=self.dataset)
        else: job_config = jobconfig
        query_job = client.query(query, job_config =job_config)  # API request
        df = query_job.result().to_dataframe()
        return df
    
    def data_types(self):

        # Query to get a list of all participants and their data types
        data_types_df = self.client_read_gbq(f"""
            SELECT 
                DISTINCT p.person_id
                , r.race
                , has_whole_genome_variant as WGS
                , has_array_data as Arr
                , has_ehr_data AS EHR
                , has_physical_measurement_data AS PM
                , has_ppi_survey_data AS PPI
                , has_fitbit AS Fitbit

            FROM `person` p
            LEFT JOIN `cb_search_person` USING(person_id)
            LEFT JOIN (SELECT DISTINCT person_id
                    , REPLACE(REPLACE(REPLACE(answer, 'What Race Ethnicity: ',''), 'Race Ethnicity ',''), 'PMI: ','') as race
                      FROM `ds_survey` WHERE question_concept_id =  1586140) r USING(person_id)

            WHERE (has_whole_genome_variant = 1 OR has_array_data = 1) """)
        data_types_df = data_types_df.rename(columns = {'Arr':"Array"})
        data_types_df['race'] = data_types_df['race'].fillna('None')
        data_types_df2 = self.race_groups(data_types_df)
        
        return data_types_df2
    
    def race_groups(self, genomic_data):
        
        genomic_races_df = genomic_data.copy()
        genomic_races_df['race'] = genomic_races_df[['person_id','race']].drop_duplicates().sort_values('race')\
                                    .groupby('person_id')['race'].transform(lambda x: ','.join(x))

        main_cats = ['AIAN','AIAN,White','Asian','Asian,White'
                     ,'Black','Black,White','Hispanic','Hispanic,White','MENA','MENA,White','White','Skip']
        genomic_races_df.loc[~genomic_races_df.race.isin(main_cats), 'race'] = 'Other'
        
        return genomic_races_df
    
    def race_counts(self, gen_data, gen_name, save = False):
        
        count_col = f'{gen_name} Participants Count'
        perc_col = f'{gen_name} Participants Percentage'
        
        gen_data_races_ct = gen_data[['person_id','race']].drop_duplicates().groupby('race').nunique()
        gen_data_races_ct.columns = [count_col]
        gen_data_races_ct[perc_col] = gen_data_races_ct[count_col]/gen_data.person_id.nunique()
        
        move_index_down = ['Other', 'Skip']
        gen_data_races_ct = gen_data_races_ct.reindex([i for i in gen_data_races_ct.index.tolist() \
                                                       if i not in move_index_down]+move_index_down)
               
        total_row = pd.DataFrame(gen_data_races_ct.sum(), columns = ['Total']).T
        total_row[count_col] = total_row[count_col].astype('int')
        
        gen_data_races_ct = pd.concat([gen_data_races_ct, total_row])
        
        if save == True:
            g = gen_name.replace(' ','_')
            fname = f'4_genomic_{g}_race_ethn.xlsx'
            gen_data_races_ct.to_excel(fname)
            args = ["gsutil", "cp", f"./{fname}", f"{self.bucket}/notebooks/data/"]
            output = subprocess.run(args, capture_output=True)
            print(output.stderr)
            print(f'\nSaved as {fname}\n')
 
        return gen_data_races_ct
    
    def datatype_combinations(self, genomic_data, gen_name, gen_set, save = False):

        ehr_set = set(genomic_data[genomic_data.EHR == 1]['person_id'])
        pm_set = set(genomic_data[genomic_data.PM == 1]['person_id'])
        fitbit_set = set(genomic_data[genomic_data.Fitbit == 1]['person_id'])
        ppi_set = set(genomic_data[genomic_data.PPI == 1]['person_id'])
        #gen_set = set(genomic_data[genomic_data[gen_name] == 1]['person_id'])



        items_list = [[gen_name ,'any '+gen_name+' data', len(gen_set)]

             , [gen_name+' and PPI' ,'any '+gen_name+' AND any PPI', len(gen_set.intersection(ppi_set))]

             , [gen_name+' and PPI and PM' ,'any '+gen_name+' AND any PPI AND any PM'
                                            , len(gen_set.intersection(ppi_set).intersection(pm_set))]

             , [gen_name+' and EHR' ,'any '+gen_name+' AND any EHR'
                                     , len(gen_set.intersection(ehr_set))]

             , [gen_name+' and PPI and EHR' ,'any '+gen_name+' AND any PPI AND any EHR'
                                             , len(gen_set.intersection(ppi_set).intersection(ehr_set))]

             , [gen_name+' and PPI and EHR and PM' ,'any '+gen_name+' AND any EHR AND any PM AND any PPI', 
                          len(gen_set.intersection(ehr_set).intersection(pm_set).intersection(ppi_set))]

             , [gen_name+' and Fitbit' ,'any '+gen_name+' AND any Fitbit', len(gen_set.intersection(fitbit_set))]

             , [gen_name+' and PPI and Fitbit' ,'any '+gen_name+' AND any PPI AND Fitbit'
                , len(gen_set.intersection(ppi_set).intersection(fitbit_set))]                 

             , [gen_name+' and PPI and PM and Fitbit' ,'any '+gen_name+' AND any PPI AND any PM AND any Fitbit',
                          len(gen_set.intersection(ppi_set).intersection(pm_set).intersection(fitbit_set))]

             , [gen_name+'  and Fitbit and PPI and EHR' 
                ,'any '+gen_name+' AND any Fitbit AND and PPI AND any EHR'
                , len(gen_set.intersection(fitbit_set).intersection(ppi_set).intersection(ehr_set))]

             , [gen_name+' and PPI and EHR and PM and Fitbit' 
                    ,'any '+gen_name+' AND any EHR AND and PM AND any PPI AND any Fitbit',
                          len(gen_set.intersection(ehr_set)\
                              .intersection(pm_set).intersection(ppi_set).intersection(fitbit_set))]
             ]

        report_df = pd.DataFrame()
        for data in items_list:
            df = pd.DataFrame({data[0]}, columns = ['Datatypes Combinations'])
            df[f'{gen_name} Participants Count'] = data[2]
            report_df = pd.concat([report_df, df])

        if save == True:
            g = gen_name.replace(' ','_')
            fname = f'4_genomic_{g}_data_comb.xlsx'
            report_df.to_excel(fname, index = False)
            args = ["gsutil", "cp", f"./{fname}", f"{self.bucket}/notebooks/data/"]
            output = subprocess.run(args, capture_output=True)
            print(output.stderr)
            print(f'\nSaved as {fname}\n')

        return report_df
    
    def genomics_report(self, genomic_data_df, gen_name):
        if 'and' in gen_name.lower():
            gen_data = genomic_data_df[(genomic_data_df[gen_name.split(' ')[0]] == 1) 
                                       & (genomic_data_df[gen_name.split(' ')[-1]] == 1)]
        else:
            gen_data = genomic_data_df[genomic_data_df[gen_name] == 1]

        gen_set = set(gen_data['person_id'])

        gen_by_race = self.race_counts(gen_data, gen_name)
        gen_data_combs = self.datatype_combinations(genomic_data_df, gen_name, gen_set)

        return gen_by_race, gen_data_combs