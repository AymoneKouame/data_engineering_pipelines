from google.cloud import bigquery
client = bigquery.Client()
import os
import subprocess
import pandas as pd
import numpy as np
from utilities import utilities as u
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class demographics_by_datatypes:    

    ## Class to get ubr flags for each person in the current dataset    
    def __init__(self):
        
        self.dataset = u().dataset
        self.bucket = u().bucket
        self.client_read_gbq = u().client_read_gbq
        
        #### to order the indexes in final table
        self.sex_index_order = ['Female', 'Male', 'Intersex','None Of These', 'Skip', 'Not Specified']

        self.race_index_order = ['White', 'Black', 'Asian', 'AIAN (American Indian or Alaska Native)'
                                 , 'MENA (Middle Eastern or North African)'
                                 , 'NHPI (Native Hawaiian or Other Pacific Islander)'
                                 , 'More Than One Population', 'None Of These', 'Skip', 'Not Specified']
        
        self.self_race_index_order = ['White', 'Black', 'Asian', 'AIAN (American Indian or Alaska Native)'
                                      , 'MENA (Middle Eastern or North African)'
                                      , 'NHPI (Native Hawaiian or Other Pacific Islander)', 'Hispanic'
                                      , 'GeneralizedMultPopulations', 'None Of These', 'Skip', 'Not Specified']


        self.ethnicity_index_order = ['Not Hispanic Or Latino', 'Hispanic Or Latino','None Of These'
                                      , 'Skip', 'Not Specified']

        self.age_index_order = ['18-29','30-39', '40-49', '50-59','60-69', '70-79', '80-89', '90+']

        self.gender_index_order = ['Woman', 'Man', 'Non Binary', 'Transgender', 'Additional Options', 'Skip', 'Not Specified']

        self.income_index_order = ['Less 10K', '10K 25K', '25K 35K', '35K 50K', '50K 75K'
                              , '75K 100K','100K 150K', '150K 200K', 'More 200K','Skip', 'Not Specified']

        self.education_index_order = ['Never Attended', 'One Through Four', 'Five Through Eight','Nine Through Eleven'
                                 , 'Twelve Or Ged', 'College One To Three' , 'College Graduate', 'Advanced Degree'
                                 , 'Skip', 'Not Specified']

        self.employment_index_order = ['Employed For Wages', 'Self Employed', 'Out Of Work One Or More'
                                       ,'Out Of Work Less Than One'
                                       ,'Student', 'Homemaker','Retired', 'Unable To Work', 'Skip', 'Not Specified']

        self.so_index_order = ['Straight', 'Bisexual', 'Gay','Prefer not to answer', 'Lesbian'
                               ,'None Of These', 'Skip', 'Not Specified']
        
        # rename certain variables
        self.new_field_names = {'native hawaiian or other pacific islander'.title():'NHPI (Native Hawaiian or Other Pacific Islander)'
                                , 'Nhpi':'NHPI (Native Hawaiian or Other Pacific Islander)'
                                
                                , 'middle eastern or north african'.title(): 'MENA (Middle Eastern or North African)'
                                , 'Mena': 'MENA (Middle Eastern or North African)'
                                
                                , 'american indian or alaska native'.title():'AIAN (American Indian or Alaska Native)'
                                , 'Aian': 'AIAN (American Indian or Alaska Native)'
                                , 'black or african american'.title():'Black'
                                , 'Noneofthese':'None Of These'
                                , 'Generalizedmultpopulations': 'GeneralizedMultPopulations'
                              }

        self.new_variable_names = {'race': 'Self-Reported Race'
                             , 'sex_at_birth':'Self-Reported Sex at Birth'
                             , 'ethnicity':'Self-Reported Ethnicity'
                             , 'gender_identity':'Self-Reported Gender Identity'
                             , 'age_group':'Age Group at CDR'
                             , 'education':'Self-Reported Educational Attainment'
                             , 'income':'Self-Reported Income'
                             , 'employment':'Self-Reported Employment'
                             , 'sexual_orientation':'Self-Reported Sexual Orientation'
                             , 'self_reported_race':'Self-Reported Categories of Demographic Descriptors'      
                                   
                             }


    def aian_pids(self):
        aian_pids = self.client_read_gbq(f'''SELECT DISTINCT person_id FROM `ds_survey`
                                    WHERE question_concept_id = 1586140 AND answer = 'What Race Ethnicity: AIAN' ''')
        return aian_pids

    def wear_pids(self):
        wear_pids = self.client_read_gbq(f'''SELECT DISTINCT person_id 
                                        FROM `wear_study` WHERE resultsconsent_wear= 'Yes' ''')
        return wear_pids

    def demographic_data(self):   
        # SQL query to get participants demographic data
        query = f"""
            SELECT
                    DISTINCT person_id
                    , LOWER(race) as race
                    , LOWER(ethnicity) as ethnicity
                    , CAST(age_at_cdr AS INTEGER) AS age_at_cdr
                    , s.sex_at_birth
                    , gi.gender_identity
                    , i.income                                
                    , edu.education   
                    , e.employment
                    , replace(so.sexual_orientation, 'none', 'none of these') as sexual_orientation
                    , srr.self_reported_race

            FROM `person`
            LEFT JOIN `cb_search_person` using(person_id)
 
            LEFT JOIN (SELECT DISTINCT person_id
                        , LOWER(REPLACE(REPLACE(answer, SPLIT(answer, ': ')[OFFSET(0)], ''), ': ','')) as sex_at_birth
                       FROM ds_survey where question_concept_id = 1585845) s using(person_id)

            LEFT JOIN (SELECT DISTINCT person_id
                        , LOWER(REPLACE(REPLACE(answer, SPLIT(answer, ': ')[OFFSET(0)], ''), ': ','')) as gender_identity
                        FROM ds_survey where question_concept_id = 1585838) gi using(person_id)

            LEFT JOIN (SELECT DISTINCT person_id
                        , LOWER(REPLACE(REPLACE(answer, SPLIT(answer, ': ')[OFFSET(0)], ''), ': ','')) as income
                        FROM ds_survey where question_concept_id = 1585375) i using(person_id)

            LEFT JOIN (SELECT DISTINCT person_id
                       , LOWER(REPLACE(REPLACE(answer, SPLIT(answer, ': ')[OFFSET(0)], ''), ': ','')) as education
                       FROM ds_survey where question_concept_id = 1585940) edu using(person_id) 

            LEFT JOIN (SELECT DISTINCT person_id
                        , LOWER(REPLACE(REPLACE(answer, SPLIT(answer, ': ')[OFFSET(0)], ''), ': ','')) as employment
                       FROM `ds_survey` where question_concept_id = 1585952) e using(person_id)

            LEFT JOIN (SELECT DISTINCT person_id
                         , LOWER(REPLACE(REPLACE(answer, SPLIT(answer, ': ')[OFFSET(0)], ''), ': ','')) as sexual_orientation
                       FROM ds_survey where question_concept_id = 1585899) so using(person_id)
                        
            LEFT JOIN (SELECT DISTINCT person_id
                        , REPLACE(REPLACE(REPLACE(self_reported_category_source_value, 'WhatRaceEthnicity_', '')
                                , 'RaceEthnicity', ''), 'PMI_', '') As self_reported_race 
                        FROM person_ext) srr using(person_id)
            """
        demographics_df = self.client_read_gbq(query).fillna('not specified')

        return demographics_df


    def all_persons(self):

        # Query to get a list of all participants and their data types
        # Query to get a list of all participants and their data types
        all_cdr_pids = self.client_read_gbq(f"""
            SELECT DISTINCT p.person_id
                , has_whole_genome_variant AS WGS 
                , has_array_data as Arr
                , has_ehr_data AS EHR
                , has_physical_measurement_data AS PM
                , has_ppi_survey_data AS PPI
                , has_fitbit AS Fitbit

            FROM `person` p
            LEFT JOIN `cb_search_person` USING(person_id) """)

        all_cdr_pids = all_cdr_pids.rename(columns = {'Arr':"Array"})
        return all_cdr_pids


    def clean_demographics(self, raw_df):
        clean_df = raw_df.copy()
        not_specifed_answers = ['prefer not to answer', 'i prefer not to answer', 'none indicated'
                                , 'no matching concept', 'PreferNotToAnswer', 'No matching concept']

        for d in clean_df.drop(['person_id', 'age_at_cdr'], axis = 1):
            clean_df[d] = clean_df[d].fillna('not specified')
            clean_df.loc[clean_df[d].isin([i for i in clean_df[d].unique()\
                                                             if 'none of these' in i]), d] = 'none of these'
            clean_df.loc[clean_df[d].isin([i for i in clean_df[d].unique() \
                                                             if i in not_specifed_answers]), d] = 'not specified'
            clean_df[d] = [i.title() for i in clean_df[d]]
            clean_df[d] = clean_df[d].replace(self.new_field_names)

            df_multiple_demog = clean_df[['person_id',d]].drop_duplicates().groupby('person_id', as_index = False).nunique()
            df_multiple_demog = df_multiple_demog[df_multiple_demog[d] >1]['person_id'].unique()  
            clean_df.loc[clean_df.person_id.isin(df_multiple_demog), d] = 'Multiple Selections'
            clean_df = clean_df.drop_duplicates()

        return clean_df

    def format_counts(self, df, n_colname, perc_colname, label_colname):

        df[n_colname] = df[n_colname].astype('int')
        df[label_colname] = df[n_colname]
        df[label_colname] = ['{:,}'.format(i).replace('.0','') for i in df[label_colname]]
        df[label_colname] = df[label_colname]+' ('+df[perc_colname+'_f'] +')'
        return df

    def cal_perc(self, df, denom_df, denom_name,  multiindex_col):

        var_col = [c for c in df.columns if 'Self-Reported' in c or 'Age Group at CDR' in c]

        n_colname = 'Count of participants '+denom_name
        perc_colname = 'Percentage'
        label_colname = 'Count (and %) of participants '+denom_name

        df_count = df.merge(denom_df)#.fillna(0)
        df_count = df_count.groupby(var_col).nunique()
        df_count.columns = [n_colname]
        df_count[n_colname] = df_count[n_colname].astype('int').fillna(0)
        df_count = df_count.reset_index()
        df_count[perc_colname] = df_count[n_colname]/denom_df.person_id.nunique()
        df_count[[n_colname, perc_colname]] = df_count[[n_colname, perc_colname]].fillna(0)

        df_count[var_col] = df_count[var_col].astype('str')
        total_df = pd.DataFrame(df_count.sum()).T
        total_df[var_col[0]] = 'TOTAL'

        df_count = df_count[var_col+[n_colname,perc_colname]].sort_values(n_colname, ascending = False)
        df_count = pd.concat([df_count, total_df])

        df_count[perc_colname+'_f'] = ['{:.2%}'.format(i) for i in df_count[perc_colname]]
        df_count[label_colname]= ['{:,}'.format(i).replace('.0','') for i in df_count[n_colname]]
        df_count[label_colname] = df_count[label_colname].astype('str')+' ('+df_count[perc_colname+'_f'].astype('str') +')'

        df_count = df_count.set_index(var_col[0]).drop([perc_colname+'_f'],axis = 1)
        df_count.columns = pd.MultiIndex.from_tuples([(multiindex_col, n_colname), (multiindex_col, perc_colname)
                                                    , (multiindex_col, label_colname)])
        return df_count

    def cal_perc_ifnotempty(self, df, denom_df, denom_name,  multiindex_col):
        if df.merge(denom_df).empty:
            DF = df[df.drop('person_id',axis = 1).columns.tolist()].drop_duplicates()
            DF = pd.concat([DF, pd.DataFrame({'TOTAL'}, columns = df.drop('person_id',axis = 1).columns.tolist())])
            
            new_index = DF.columns[0]
            DF[(multiindex_col, f'Count of participants with {denom_name}')] = 0
            DF[(multiindex_col, f'Percentage {denom_name}')] = 0
            DF[(multiindex_col, f'Count (and %) of participants {denom_name}')] = "0 (0.00%)"
            DF = DF.set_index(new_index)
            DF.columns = pd.MultiIndex.from_tuples(DF.columns)
        else:
            DF = self.cal_perc(df = df, denom_df = denom_df, denom_name = denom_name, multiindex_col= multiindex_col)
        return DF
        
        
    def get_counts(self, df, demographic, denom_df, denom_name, multiindex_col):

        df.loc[df[demographic].isnull(), demographic] = 'Not Specified'
               
        if demographic =='age_at_cdr':
            # create age group similar to that in data browser
            age_bins = [0,29,39,49,59,69,79,89,200]
            age_labels = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90+"]
            df['age_group'] = pd.cut(df[demographic], bins=age_bins, labels=age_labels, include_lowest=True)
            df = df.drop(demographic, axis = 1).drop_duplicates()

        elif demographic =='self_reported_race': df = df
            
        else:
            df_multiple_demog = df[['person_id',demographic]].drop_duplicates()\
                                        .groupby('person_id', as_index = False).nunique()
            df_multiple_demog = df_multiple_demog[df_multiple_demog[demographic] >1]['person_id'].unique()  
            df.loc[df.person_id.isin(df_multiple_demog), demographic] = 'Multiple Selections'
            df = df.drop_duplicates()

        df = df.rename(columns = self.new_variable_names)
        df = self.cal_perc_ifnotempty(df = df, denom_df = denom_df, denom_name = denom_name, multiindex_col= multiindex_col)

        return df

    def combined_datatypes(self, demographic, demographics_df, index_order, all_cdr_pids):

        # Denominators and Stratificators    
        overall_df = all_cdr_pids[['person_id']].drop_duplicates()
        df_plot = self.get_counts(df = demographics_df[['person_id',demographic]]
                                 , demographic = demographic, denom_df = overall_df
                                 , denom_name = 'in CDR', multiindex_col = 'OVERALL')

        for dtype in ['EHR','PPI','PM','Fitbit','WGS','Array']:
            denom_df = all_cdr_pids[all_cdr_pids[dtype] ==1][['person_id']].drop_duplicates()
            df = self.get_counts(df = demographics_df[['person_id',demographic]], demographic = demographic
                                , denom_df = denom_df, denom_name = f'with {dtype} data', multiindex_col = dtype.upper())
            df_plot = pd.concat([df_plot, df], axis =1)

        if 'Multiple Selections' in df_plot.index:
            df_plot = df_plot.reindex(index_order+['Multiple Selections']+['TOTAL'])
        else:
            df_plot = df_plot.reindex(index_order+['TOTAL'])

        perc_col = [c for c in df_plot.columns if 'Percentage' in c[0] or 'Percentage' in c[1]]
        n_col = [c for c in df_plot.columns if 'Count of participants' in c[0] or 'Count of participants' in c[1]]
        label_col = [c for c in df_plot.columns if '(and %)' in c[0] or '(and %)' in c[1]]

        df_plot[n_col] = df_plot[n_col].fillna(0).astype('int')
        df_plot[perc_col] = df_plot[perc_col].fillna(0)
        df_plot[label_col] = df_plot[label_col].fillna('0')

        # Separating the dataframe into one for table display and one for plotting
        df = df_plot[[c for c in df_plot.columns if 'Count of participants' not in c[0] and 'Count of participants' not in c[1]]]
        df = df[[c for c in df.columns if 'Percentage' not in c[0] and 'Percentage' not in c[1]]]

        return df

    def order_multiindex_df(self, df, order_keys):
        ordered_cols = []
        for k in order_keys: ordered_cols = ordered_cols+[c for c in df.columns if k in c]
        ordered_df = df[ordered_cols]
        ordered_df.columns = pd.MultiIndex.from_tuples(ordered_df.columns)
        return ordered_df


    def combine_dataframes(self, demographic, demographics_df, index_order, all_cdr_pids, save = False):

        aian_pids_df = self.aian_pids()
        wear_pids_df = self.wear_pids()

        whole_cdr_df = self.combined_datatypes(demographic= demographic
                                              , demographics_df = demographics_df
                                              , index_order = index_order
                                              , all_cdr_pids = all_cdr_pids)

        aian_df = self.combined_datatypes(demographic= demographic
                                         , demographics_df = demographics_df.merge(aian_pids_df)
                                         , index_order = index_order
                                         , all_cdr_pids = all_cdr_pids.merge(aian_pids_df)
                                        )

        wear_df = self.combined_datatypes(demographic= demographic
                                         , demographics_df = demographics_df.merge(wear_pids_df)
                                         , index_order = index_order
                                         , all_cdr_pids = all_cdr_pids.merge(wear_pids_df)
                                        )

        whole_cdr_df.columns = [(c[0], 'Whole CDR (count and %)') for c in whole_cdr_df.columns]
        aian_df.columns = [(c[0], 'AIAN (count and %)') for c in aian_df.columns]
        wear_df.columns = [(c[0], 'WEAR (count and %)') for c in wear_df.columns]



        concat_df = pd.concat([whole_cdr_df, aian_df, wear_df], axis = 1)
        combined_df = self.order_multiindex_df(concat_df, order_keys = ['OVERALL', 'PPI','PM', 'FITBIT', 'WGS', 'ARRAY',  'EHR'])

        if save == True:
            fname = f'2_demographic_{demographic}.xlsx'
            u().write_to_csv_excel(combined_df, filename = fname)

        return combined_df