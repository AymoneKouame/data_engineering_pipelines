# Functions and classes to perform complex data collection, wrangling, formatting and export stats reports/visulaizations of EHR data

from google.cloud import storage
import os

# For Data Manipulation
import numpy as np
import pandas as pd
import math
import statistics
import datetime 

# For Data Visualization
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

###############
dataset = os.getenv('WORKSPACE_CDR')

# Function to save a file to bucket
def upload_blob(source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    bucket_name = os.getenv('WORKSPACE_BUCKET').replace('gs://','')
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name = bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


############################################## DATA COLLECTION #####################################################

def historic_ehr(dataset = dataset):
        
    def ehr_by_domain(domain, start_date_field, end_date_field, dataset, table, table_id, cutoff):

        concept_id = domain.lower()+'_concept_id'
            #print(domain, start_date_field, end_date_field, table, table_id, concept_id)
        df = pd.read_gbq(f''' SELECT DISTINCT person_id
                                    , '{domain}' as ehr_domain
                                    , src_id AS ehr_site
                                    , vocabulary_id as vocabulary
                                    , MIN({start_date_field}) AS start_date
                                    , CASE WHEN MAX({end_date_field}) IS NULL THEN "{cutoff}"
                                        ELSE MAX({end_date_field}) END AS end_date

                                FROM `{dataset}.{table}`
                                LEFT JOIN `{dataset}.{table}_ext` USING({table_id})
                                JOIN `{dataset}.concept` on concept_id = {concept_id}
                                WHERE LOWER(src_id) LIKE 'ehr site%'
                                GROUP BY 1,2,3,4
                                ''')

        return df
        ##########################################################################

    # setting variables
    tables = ['measurement', 'condition_occurrence','device_exposure','drug_exposure'
                  ,'observation','procedure_occurrence', 'visit_occurrence']
    start_end_date_fields = [['measurement_date', 'measurement_date']
                                 , ['condition_start_date','condition_end_date']
                                 , ['device_exposure_start_date','device_exposure_end_date']
                                 , ['drug_exposure_start_date','drug_exposure_end_date']
                                 , ['observation_date','observation_date']
                                 , ['procedure_date','procedure_date']
                                 , ['visit_start_date','visit_end_date']
                                ]
    domains = ['Measurement', 'Condition','Device','Drug', 'Observation','Procedure', 'Visit']
    table_ids = ['measurement_id', 'condition_occurrence_id','device_exposure_id', 'drug_exposure_id'
                     ,'observation_id', 'procedure_occurrence_id','visit_occurrence_id']


    ehr_cutoff_date = pd.read_gbq(f'''SELECT ehr_cutoff_date  FROM `{dataset}._cdr_metadata`''')
    ehr_cutoff =str(ehr_cutoff_date.ehr_cutoff_date[0])   
        
    historic_ehr_raw = pd.DataFrame()
        
    for i in range(len(tables)):
        df = ehr_by_domain(domain = domains[i]
                                , start_date_field = start_end_date_fields[i][0]+'time'
                                , end_date_field = start_end_date_fields[i][1]+'time'
                                , dataset = dataset
                                , table = tables[i], table_id = table_ids[i]
                                , cutoff = ehr_cutoff)
        historic_ehr_raw = pd.concat([historic_ehr_raw, df])

    historic_overall = historic_ehr_raw.copy()
    historic_overall['ehr_domain'] = 'Any EHR'
    historic_overall = historic_overall.drop_duplicates()#.groupby(['person_id','ehr_domain'], as_index= False).min()
    historic_ehr = pd.concat([historic_ehr_raw, historic_overall])

    return historic_ehr
    
def demographics(dataset = dataset): 

    def clean_demographic_data(demog_df, dataset = dataset):

        def clean(df, demographic):
            df[demographic] = df[demographic].fillna('Not Specified')

            df_multiple_demog = df[['person_id',demographic]].groupby('person_id', as_index = False).nunique()
            df_multiple_demog = df_multiple_demog[df_multiple_demog[demographic] >1]['person_id'].unique()  
            df.loc[df.person_id.isin(df_multiple_demog), demographic] = 'Multiple Selections'
            df = df.drop_duplicates()

            df.loc[df[demographic].str.contains(': '), demographic] = \
                        [i.split(': ')[1] for i in df.loc[df[demographic].str.contains(': '), demographic]]

            df[demographic] = [i.title() for i in df[demographic]]


            new_field_names = {'race ethnicity none of these'.title():'None Of These'
                                    , "sex at birth none of these".title():'None Of These'
                                    , "none".title():'None Of These'
                                    , 'nhpi'.title():'NHPI (Native Hawaiian and Other Pacific Islander)'
                                    , 'mena'.title(): 'MENA (Middle Eastern and North African)'
                                    #, 'pmi: skip'.title(): 'Skip'
                                   # , 'no matching concept'
                                    , 'prefer not to answer'.title(): 'Not Specified'
                                    , 'i prefer not to answer'.title(): 'Not Specified'
                                    , 'no matching concept'.title(): 'Not Specified'
                                    , 'additional options'.title(): 'Not Specified'
                                    , 'not man only, not woman only, prefer not to answer, or skipped'.title():'Not Specified'
                                  }
            df[demographic] = df[demographic].replace(new_field_names)
            return df[demographic]

        demog_clean_df = demog_df.copy()
        for d in demog_df.drop('person_id',1).columns:
            if d == 'age_at_cdr':
                # create age groups           
                demog_clean_df['age_group'] = pd.cut(demog_clean_df[d]
                                                         , bins= [18,29,39,49,59,69,79,89,int(demog_clean_df[d].max())]
                                                         , labels=["18-29", "30-39", "40-49", "50-59"
                                                                   , "60-69", "70-79", "80-89", "90+"]
                                                         , include_lowest=True)

                demog_clean_df = demog_clean_df.drop(d,1)

            else:
                if 'age' not in d: 
                    demog_clean_df[d] = clean(df = demog_df, demographic = d)

        return demog_clean_df.drop_duplicates()

  
    # SQL query to get participants demographic data
    query = f"""
            SELECT
                DISTINCT person_id
                , r.race
                , LOWER(e.concept_name) AS ethnicity
                , CAST(age_at_cdr AS INTEGER) AS age_at_cdr
                , s.sex_assigned_at_birth
                , gender_identity
                , annual_income                                
                , educational_attainment   
                , employment  

            FROM `{dataset}.person`
            JOIN `{dataset}.concept` e on e.concept_id = ethnicity_concept_id
            LEFT JOIN `{dataset}.cb_search_person` using(person_id)
            
            LEFT JOIN (SELECT DISTINCT person_id, LOWER(concept_name) as race
                      FROM `{dataset}.observation`
                      JOIN `{dataset}.concept` on concept_id = value_source_concept_id
                      WHERE observation_source_concept_id = 1586140) r using(person_id)
                      
            LEFT JOIN (SELECT DISTINCT person_id, LOWER(concept_name) as sex_assigned_at_birth
                      FROM `{dataset}.observation`
                      JOIN `{dataset}.concept` on concept_id = value_source_concept_id
                      WHERE observation_source_concept_id = 1585845) s using(person_id)

            LEFT JOIN (SELECT DISTINCT person_id, LOWER(concept_name) as annual_income
                      FROM `{dataset}.observation`
                      JOIN `{dataset}.concept` on concept_id = value_source_concept_id
                      WHERE observation_source_concept_id = 1585375) using(person_id)

            LEFT JOIN (SELECT DISTINCT person_id, LOWER(concept_name) as gender_identity
                      FROM `{dataset}.observation`
                      JOIN `{dataset}.concept` on concept_id = value_source_concept_id
                      WHERE observation_source_concept_id = 1585838) using(person_id)

            LEFT JOIN (SELECT DISTINCT person_id, LOWER(concept_name) as educational_attainment
                      FROM `{dataset}.observation` 
                      JOIN `{dataset}.concept` on concept_id = value_source_concept_id
                      where observation_source_concept_id = 1585940) using(person_id) 

            LEFT JOIN (SELECT DISTINCT person_id, LOWER(concept_name) as employment
                      FROM `{dataset}.observation` 
                      JOIN `{dataset}.concept` on concept_id = value_source_concept_id
                      WHERE observation_source_concept_id = 1585952) using(person_id)
                      
            """
    raw_demographics_df = pd.read_gbq(query)
    demographics_df = clean_demographic_data(raw_demographics_df)

    return demographics_df
    
def ubr(dataset = dataset):
    
    # Function to query and calculate UBR categories 

    nulls = ['Null','No matching concept','PMI_Skip','PMI_PreferNotToAnswer', 'AoUDRC_NoneIndicated']
    nulls_sql = tuple(nulls)

    query = f"""
            SELECT
                DISTINCT person_id
                , CASE WHEN (race_source_value IS NOT NULL 
                                AND race_source_value NOT IN ('WhatRaceEthnicity_White')
                                AND race_source_value NOT IN {nulls_sql})
                            OR (ethnicity_source_value = 'WhatRaceEthnicity_Hispanic') 
                        THEN 'yes' else 'no' END as ubr_race_ethnicity

                , CASE WHEN age_at_consent <18 OR age_at_consent >=65 THEN 'yes' ELSE 'no' END as ubr_age

                , CASE WHEN sex_at_birth_source_value IN ('SexAtBirth_Intersex','SexAtBirth_SexAtBirthNoneOfThese')
                        THEN 'yes' ELSE 'no' END as ubr_sex_assigned_at_birth

                , CASE WHEN income IN ('AnnualIncome_less10k', 'AnnualIncome_10k25k')
                        THEN 'yes' ELSE 'no' END as ubr_annual_income

                , CASE WHEN education IN ('HighestGrade_NineThroughEleven', 'HighestGrade_FiveThroughEight'
                                        , 'HighestGrade_OneThroughFour','HighestGrade_NeverAttended')
                        THEN 'yes' ELSE 'no' END as ubr_educational_attainment   

                , CASE WHEN 
                        ((gender_source_value IS NOT NULL
                            AND gender_source_value NOT IN {nulls_sql})
                            AND gender_source_value NOT IN ('GenderIdentity_Woman', 'GenderIdentity_Man'))

                        OR (gender_source_value = 'GenderIdentity_Woman' 
                            AND sex_at_birth_source_value !='SexAtBirth_Female'
                            AND gender_source_value IS NOT NULL
                            AND gender_source_value NOT IN {nulls_sql}
                            AND sex_at_birth_source_value IS NOT NULL
                            AND sex_at_birth_source_value NOT IN {nulls_sql}
                            )
                        OR (gender_source_value = 'GenderIdentity_Man' 
                            AND sex_at_birth_source_value !='SexAtBirth_Male'
                            AND gender_source_value IS NOT NULL
                            AND gender_source_value NOT IN {nulls_sql}
                            AND sex_at_birth_source_value IS NOT NULL
                            AND sex_at_birth_source_value NOT IN {nulls_sql}
                            )
                        OR (sexual_orientation IS NOT NULL 
                            AND sexual_orientation NOT IN {nulls_sql}
                            AND sexual_orientation NOT IN ('SexualOrientation_Straight')) 
                        THEN  'yes' ELSE 'no' END as ubr_sexual_and_gender_minorities



            FROM `{dataset}.person`
            LEFT JOIN `{dataset}.cb_search_person` using(person_id)
            LEFT JOIN (SELECT DISTINCT person_id, value_source_value as income
                      FROM `{dataset}.observation` where observation_source_concept_id = 1585375) using(person_id)

            LEFT JOIN (SELECT DISTINCT person_id, value_source_value as education
                      FROM `{dataset}.observation` where observation_source_concept_id = 1585940) using(person_id) 

            LEFT JOIN (SELECT DISTINCT person_id, value_source_value as sexual_orientation
                      FROM `{dataset}.observation` where observation_source_concept_id = 1585899) using(person_id)
                      
        """
    ubr_df = pd.read_gbq(query)

    # When multiple sexual orientations or gender identities are chosen
    ## assign 'yes' to SGM if the participant meets the UBR SGM criteria for at least one of the 
    ## sexual orientations or gender identities chosen
    for c in ubr_df.drop(['person_id'],1).columns:
        multiple_ubrs = ubr_df[['person_id',c]].groupby('person_id').nunique()
        multiple_ubrs_pids = multiple_ubrs[multiple_ubrs[c] >1].index.unique()
        ubr_df.loc[ubr_df.person_id.isin(multiple_ubrs_pids), c] = 'yes'

    ubr_df = ubr_df.drop_duplicates()
    ubr_df['ubr_overall'] = (np.where((ubr_df['ubr_race_ethnicity']=='yes') 
                                          | (ubr_df['ubr_age']=='yes') 
                                          | (ubr_df['ubr_sex_assigned_at_birth']=='yes')
                                          | (ubr_df['ubr_sexual_and_gender_minorities']=='yes') 
                                          | (ubr_df['ubr_annual_income']=='yes') 
                                          | (ubr_df['ubr_educational_attainment']=='yes') ,'yes', 'no')
                                       )

    return ubr_df



############################################## DATA WRANGLING AND FORMATTING #####################################################
class historic_ehr_by_domain:

    def __init__(self, historic_ehr_df, denominator, dataset = dataset):
        self.dataset = dataset
        self.historic_ehr_df = historic_ehr_df
        self.denominator = denominator
                
    ############################# 1. Historic EHR data by Domains over time, dataframe #############################            
    def years_of_data(self, which):
        df = self.historic_ehr_df.reset_index(drop = True)
        df = df[['person_id','ehr_domain','start_date', 'end_date']].drop_duplicates()
        ###
        end_date_max = df[['person_id','ehr_domain','end_date']].drop_duplicates()\
                            .groupby(['person_id','ehr_domain'], as_index = False).max()

        start_date_min = df[['person_id','ehr_domain','start_date']].drop_duplicates()\
                            .groupby(['person_id','ehr_domain'], as_index = False).min()

        df1 = df.drop(['start_date','end_date'],1).merge(start_date_min, 'left')\
                                .merge(end_date_max, 'left').drop_duplicates()

        df1['days_diff'] = df1['end_date'] - df1['start_date']
        df1['days_diff'] = [abs(i.days) for i in df1['days_diff']] 


        df1.loc[df1['days_diff'] == 0, 'days_diff'] = 1
        
        ###
        df_years_of_data = df1[['person_id','ehr_domain','days_diff']]\
                                .groupby(['person_id','ehr_domain'], as_index = False).sum()
        df_years_of_data['years_of_data'] = df_years_of_data['days_diff']/365
        df_years_of_data = df_years_of_data.sort_values('years_of_data')
        years_of_data_per_pid = df_years_of_data.drop('days_diff',1)

        
    ###############################    
        
        def stats_df(years_of_data_per_pid):
            historic_ehr_years_stats = years_of_data_per_pid.groupby(['ehr_domain']).agg({'person_id':'nunique', 'years_of_data'\
                                                                      :['mean','median', 'min','max','std']})

            historic_ehr_years_stats.columns = [c[1] for c in historic_ehr_years_stats.columns]
            historic_ehr_years_stats[['mean','median','min','max', 'std']] = \
            historic_ehr_years_stats[['mean','median','min','max', 'std']].apply(lambda x: round(x,2)).astype('float64')

            historic_ehr_years_stats[['nunique','mean','median','min','max', 'std']] = \
                            format_numbers(historic_ehr_years_stats[['nunique','mean','median','min','max', 'std']])
            historic_ehr_years_stats['min'] = historic_ehr_years_stats['min'].replace('0%','0')

            name = 'Years of EHR Data Available'
            historic_ehr_years_stats.columns = pd.MultiIndex.from_tuples(
                                                    [('','N Participants')
                                                    ,(f"{name}",'Mean')
                                                    ,(f"{name}",'Median')
                                                    ,(f"{name}",'Minimum')
                                                    ,(f"{name}",'Maximun')
                                                    ,(f"{name}",'Standard Deviation')
                                                    ])

            return style_df(historic_ehr_years_stats)

        def boxplot(years_of_data_per_pid, w = 16, l = 8):

            df_plot = years_of_data_per_pid.groupby(['ehr_domain','years_of_data'], as_index = False).nunique()

            current_cdr = dataset.split('.')[1]       
 
            sns.set(rc={'figure.figsize':(w, l)})
            g = sns.boxplot(data = df_plot
                            , x = 'ehr_domain', y = 'years_of_data', hue = 'ehr_domain'
                            , medianprops={"color": "red"}, meanprops={"color": "coral"}
                            , showmeans = True, dodge=False)

            g.get_legend().remove()

            plt.suptitle(f'''\nIn the current CDR ({current_cdr}) for N Participants = {'{:,}'.format(self.denominator)}\n'''
                         , size = 16,style='italic')
            plt.title('\nYears of EHR Data Available\n\n\n', size = 20)
            plt.xlabel('EHR Domains', size = 16)
            plt.ylabel('N Years of Data', size = 16)
            plt.xticks(size = 14)
            plt.yticks(size = 14)

            plt.show()
            
        if which == 'stats':
            figure = stats_df(years_of_data_per_pid)
        else:
            if which == 'boxplot':
                figure = boxplot(years_of_data_per_pid)
                
        return figure
    
    ############################# 2. Historic EHR data by Domains over time, plot #############################            
    def lineplot(self, add_perc_text = 'yes', rotate_x = 0, plot_start_year = 2010#, save_plot = 'yes'
                , w = 16, l = 9):
                
        def pid_cumcount_by_vars(df, cum_count_by):

            cummulative_counts_df = pd.DataFrame()

            for d in df[cum_count_by[0]].unique():
                #print('\n'+d)
                DF = df[df[cum_count_by[0]] == d][['person_id']+cum_count_by].sort_values(cum_count_by, ascending = True)
                DF['cumcount'] = (~DF['person_id'].duplicated()).cumsum()
                cum_df = DF.drop('person_id',1).drop_duplicates().reset_index(drop = True)
                cum_df[cum_count_by[0]] = d
                cum_df = cum_df.sort_values(cum_count_by).groupby(cum_count_by, as_index = False).last() 

                cummulative_counts_df = pd.concat([cummulative_counts_df, cum_df])

            cummulative_counts_df['cumcount'] = cummulative_counts_df['cumcount'].astype('int')

            return cummulative_counts_df

        #############################################

        #Transform
        historic_ehr_cum = self.historic_ehr_df.copy()
        historic_ehr_cum['start_year'] = [i.year for i in historic_ehr_cum['start_date']]
        historic_ehr_cum = pid_cumcount_by_vars(df  = historic_ehr_cum[['person_id','ehr_domain','start_year']].drop_duplicates()
                                                , cum_count_by = ['ehr_domain','start_year'])
        historic_ehr_cum['cum%'] = round((historic_ehr_cum['cumcount']/self.denominator)*100,2)

        historic_ehr_cum_after2000 = historic_ehr_cum[historic_ehr_cum.start_year >= plot_start_year]

        df_plot = historic_ehr_cum_after2000.drop(['cumcount'],1)\
                    .pivot(index = ['start_year'], columns = ['ehr_domain']).fillna(0)#.reset_index()
        df_plot.columns = [c[1] for c in df_plot.columns]

        current_cdr = dataset.split('.')[1]
        fig_caption = f'''\n\nThis plot represents EHR data availability in the current CDR ({current_cdr}). 
        The denominator for the percentages is the Total Number of Participants in the current CDR, N = {'{:,}'.format(self.denominator)}.
        For better visibility, the plot only displays EHR data growth starting in 2010. '''

        #plot
        plt.figure(figsize=(w,l), tight_layout=True)
        plt.plot(df_plot, 'o-', linewidth=2)

        plt.xticks(df_plot.index, size = 13, rotation = rotate_x)
        plt.yticks(range(int(historic_ehr_cum_after2000['cum%'].min())
                         , int(historic_ehr_cum_after2000['cum%'].max()), 5),size = 13)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(decimals=0))
        plt.xlabel('Year\n\n', size = 16)
        plt.ylabel('% of Participants with EHR data', size = 16)
        plt.title('\nHistorical Availability of EHR Records\n', size = 21)
        plt.xticks(size = 14)
        plt.yticks(size = 14)
        plt.legend(title='EHR Domains', title_fontsize = 14, labels=df_plot.columns, fontsize = 14)
        plt.text(statistics.median(df_plot.index), -16
             , fig_caption, verticalalignment='bottom',style='italic'
             , horizontalalignment='center', color = 'black', size = 14)


        if add_perc_text.lower() == 'yes':
            min_year = df_plot.index.min()
            max_year = df_plot.index.max()

            for d in ['Any EHR']:
                plt.text(min_year+0.5, df_plot[d][min_year]+2
                         , d+' ('+str(int(df_plot[d][min_year]))+'%)',verticalalignment='center'
                         ,horizontalalignment='right', color = 'black', size = 12, rotation = 13)
                plt.text(max_year-0.0001, df_plot[d][max_year]+1
                         , d+' ('+str(int(df_plot[d][max_year]))+'%)',verticalalignment='bottom'
                         ,horizontalalignment='center', color = 'black', size = 12)

            for d in ['Condition', 'Device', 'Drug'#,'Measurement'
                      ,'Observation'#,'Procedure'
                      ,'Visit']:
                plt.text(min_year-0.1, df_plot[d][min_year]-0.6
                         , str(int(df_plot[d][min_year]))+'%',verticalalignment='center'
                         ,horizontalalignment='right', color = 'black', size = 12, rotation = 13)

            for d in ['Condition', 'Device', 'Drug'#,'Measurement'
                      ,'Observation'#,'Procedure','Visit'
                     ]:

                plt.text(max_year+0.2, df_plot[d][max_year]-1
                         , str(int(df_plot[d][max_year]))+'%',verticalalignment='bottom'
                         ,horizontalalignment='center', color = 'black', size = 12)

        plt.show()

############################################## ADDITIONAL DATA FORMATTING #####################################################
def format_numbers(df):
    # format the counts and percentage columns
    formated_df = df.copy()
    for col in formated_df.columns:
        if formated_df[col].dtype == 'int64':
            less_than_20 = formated_df.loc[formated_df[col] <20, col].values
            formated_df.loc[formated_df[col].astype('int64') <20, col]= '<20'
            
            formated_df.loc[formated_df[col] != '<20', col] = \
                    ['{:,}'.format(i) for i in formated_df.loc[formated_df[col]!= '<20', col]]           
            
        else:
            if (formated_df[col].dtype == 'float64') & np.all(formated_df[col].values<=1):
                formated_df[col] = ['{:.0%}'.format(i) for i in formated_df[col]]
            
    return formated_df

def combine_count_and_perc(df, count_perc_cols_dic, drop = False):    
    df_comb = df.copy()
    for count_col in count_perc_cols_dic:
        perc_col = count_perc_cols_dic[count_col]
        df_comb[count_col+'_str'] = df_comb[count_col]
        df_comb[perc_col+'_str'] = df_comb[perc_col]
        df_comb[[count_col+'_str', perc_col+'_str']] = format_numbers(df_comb[[count_col+'_str', perc_col+'_str']])
        df_comb.loc[df_comb[count_col+'_str'].str.contains('<20'), perc_col+'_str']= '-'
        df_comb.loc[(~df_comb[count_col+'_str'].str.contains('<20')) & (df_comb[perc_col] <0.01)
                    , perc_col+'_str']= '<1%'
        
        new_col_name = count_col+' (and %)'
        df_comb[new_col_name] = df_comb[count_col+'_str']+' ('+df_comb[perc_col+'_str']+')'
        
        if drop == True:
            df_comb = df_comb.drop([count_col, count_col+'_str', perc_col, perc_col+'_str'],1)
    return df_comb


####

generic_demog_cat = ['Multiple Selections', 'None Of These', 'Skip', 'Not Specified','TOTAL']
sex_index_order = ['Female', 'Male', 'Intersex']+generic_demog_cat

race_index_order = ['White', 'Black', 'Hispanic','Asian', 'MENA (Middle Eastern and North African)'
                     , 'NHPI (Native Hawaiian and Other Pacific Islander)', 'More Than One Population'
                     ]+generic_demog_cat

ethnicity_index_order = ['Not Hispanic Or Latino', 'Hispanic Or Latino']+generic_demog_cat

age_index_order = ['18-29','30-39', '40-49', '50-59','60-69', '70-79', '80-89', '90+', 'TOTAL']

gender_index_order = ['Woman', 'Man', 'Non Binary', 'Transgender']+generic_demog_cat

income_index_order = ['Less 10K', '10K 25K', '25K 35K', '35K 50K', '50K 75K'
                      , '75K 100K','100K 150K', '150K 200K', 'More 200K']+generic_demog_cat

education_index_order = ['Never Attended', 'One Through Four'
                         , 'Five Through Eight','Nine Through Eleven'
                         , 'Twelve Or Ged', 'College One To Three' 
                         , 'College Graduate', 'Advanced Degree'
                         ]+generic_demog_cat

employment_index_order = ['Employed For Wages', 'Self Employed'
                          , 'Out Of Work One Or More','Out Of Work Less Than One'
                          ,'Student', 'Homemaker','Retired', 'Unable To Work'
                          ]+generic_demog_cat

sorting_dict = {#'pm':pm_index_order
                'sex_assigned_at_birth': sex_index_order
                ,'race':race_index_order
                ,'ethnicity': ethnicity_index_order
                , 'age_group':age_index_order
                , 'gender_identity':gender_index_order
                ,'annual_income':income_index_order
                ,'educational_attainment':education_index_order
                ,'employment':employment_index_order
               }

##################
def format_index_or_col(df, d_type, what = 'index', n=0, sorting_dict = sorting_dict):
    
    column_name = what if what in df.columns else df.columns[0]

    items_to_sort_dict = {'index':df.index, column_name:df[column_name], 'columns':df.columns}
    items_to_sort = items_to_sort_dict[what].unique()

    sorter = []
    for p in sorting_dict[d_type]:
        sorter = sorter+[i for i in items_to_sort if p.lower() in i.lower() and i not in sorter]
    
    donot_title = ['MENA (Middle Eastern and North African)'
                   ,'NHPI (Native Hawaiian and Other Pacific Islander)', 'TOTAL']    
    if what.lower() == 'index':
        df_ordered = df.copy().reindex(sorter)
        df_ordered.index = [i.title() for i in df_ordered.index if 'str' in str(type(i)) and i not in donot_title]+\
                            [i for i in df_ordered.index if 'str' in str(type(i)) and i in donot_title]
                
    else:
        if what.lower() == column_name:
            #print('COLNAME')
            df_ordered = df.copy()
            df_ordered[column_name] = pd.Categorical(df_ordered[column_name], categories=sorter, ordered=True)
            df_ordered = df_ordered.sort_values(column_name)
        
        else:
            if what.lower() == 'columns':
                df_ordered = df.copy()[sorter]
                df_ordered.columns = [i.title() for i in df_ordered.columns]
            
    return df_ordered


def style_df(df, note = None):
    
    styled_df = df.copy()
    styled_df.index.names = [i.replace('_',' ').title() for i in styled_df.index.names]
    
    if np.all([len(i) ==1 for i in styled_df.columns.values]):
        styled_df.columns = [i.replace('_',' ').title() for i in styled_df.columns]
        
    styled_df = styled_df.astype('str')
    
    # Style the datframe for display
    s = styled_df.astype('str').style
    s.set_table_styles([
            {'selector': 'th','props': 'font-size: 1.1em'}, 
            {'selector': 'th.col_heading', 'props': 'text-align: center'},
            {'selector': 'th.col_heading.level0', 'props': 'font-size: 1.2em;'},
            {'selector': 'td', 'props': 'text-align: right; font-weight: normal; font-size: 1.2em'},
            
            ], overwrite=False)
    
    for cols in df.columns:
        s.set_table_styles({
                cols: [{'selector': 'th', 'props': 'border-left: 0.1px solid #808080'},
                       {'selector': 'td', 'props': 'border-left: 0.1px solid #808080'}]
            }, overwrite=False, axis=0)

    display(s)
    notes = {'note1':'''\n* blabla.'''
             ,'note2': '''* .'''}

    if note:
        print(notes[note])
        
############################################## FINAL REPORTS #####################################################

class ehr_by:
    
    def __init__(self, historic_ehr_df, demographics_df, ubr_df):
        self.historic_ehr_df = historic_ehr_df
        self.demographics_df = demographics_df
        self.ubr_df = ubr_df
    
    
    def ehr_by_var(self, which, var = None):
        
        def ehr_by_var_code(var_df, var, ehr_domain_col = 'ehr_domain', historic_ehr_df = self.historic_ehr_df):        
            var_df = var_df[['person_id', var]].drop_duplicates()
            all_participants = var_df.copy()
            all_participants[ehr_domain_col] = 'All Pids'
            denom = int(all_participants.person_id.nunique())

            ehr_domain_df = historic_ehr_df[['person_id','ehr_domain']].drop_duplicates()

            ehr_by_var_df = ehr_domain_df.merge(var_df, 'left')
            ehr_by_var_df[ehr_domain_col] = ehr_by_var_df[ehr_domain_col].fillna('No Data')

            ehr_by_var_df = pd.concat([ehr_by_var_df, all_participants])
            ehr_by_var_df = ehr_by_var_df\
                                    .groupby([var, ehr_domain_col]).nunique().reset_index()\
                                    .pivot(index = [var], columns = [ehr_domain_col])
            ehr_by_var_df.columns = [c[1] for c in ehr_by_var_df.columns]

            ## add columns totals
            col_total = pd.DataFrame(ehr_by_var_df.sum()).T

            col_total[var]= 'TOTAL'; col_total = col_total.set_index(var)
            ehr_by_var_df = pd.concat([ehr_by_var_df, col_total])

            ehr_by_var_df.columns = [i.title().replace('Ehr','EHR') for i in ehr_by_var_df.columns]

            # Adding percentages for the counts
            columns_dict = dict()
            ehr_by_var_df = ehr_by_var_df.fillna(0).astype('int64')
            for n_col in ehr_by_var_df.columns:
                #denom = int(denom_df.loc[n_col]['denom'])
                ehr_by_var_df[n_col+' %'] = ehr_by_var_df[n_col]/denom
                columns_dict[n_col] = n_col+' %'

            ehr_by_var_df.index.name =  ehr_by_var_df.index.name.title()
            ehr_by_var_df_display = combine_count_and_perc(ehr_by_var_df, columns_dict, drop = True)

            if 'UBR' not in ehr_by_var_df_display.index:
                ehr_by_var_df_display = format_index_or_col(ehr_by_var_df_display, d_type = var, what = 'index')
                ehr_by_var_df_display.index.names = ['Self-reported '+var.title()]

            else:
                if 'UBR' in ehr_by_var_df_display.index:            
                    ehr_by_var_df_display['Diversity Categories'] = var.replace('_',' ').title()
                    ehr_by_var_df_display = ehr_by_var_df_display.reset_index()\
                                            .rename(columns = {var.title(): ' '})

            return ehr_by_var_df_display

        if which.lower() == 'demographic':

            ehr_by_var_df = ehr_by_var_code(var = var, var_df = self.demographics_df)

        else:
            if which.lower() == 'ubr':
                #3 ADD TO THE FUNCTIONS
                ubr_df1 = self.ubr_df.replace({'yes':'UBR','no':'RBR'})
                for col in ubr_df1.drop('person_id',1).columns:
                    new_col = col.replace('ubr_','')
                    ubr_df1 = ubr_df1.rename(columns = {col:new_col})

                ehr_by_var_df = pd.DataFrame()
                for var in ubr_df1.drop('person_id',1).columns:
                    df0 = ehr_by_var_code(var_df = ubr_df1, var = var)
                    df = df0.reindex(df0.index.tolist()+[' ']+[' '])
                    df = df.fillna('')
                    ehr_by_var_df = pd.concat([ehr_by_var_df,df])

                ehr_by_var_df = ehr_by_var_df.set_index(['Diversity Categories',' '])

        return ehr_by_var_df

###################################

class by_site:
    
    def __init__(self, historic_ehr_df, denominator, dataset = dataset):
        self.historic_ehr_df = historic_ehr_df
        self.dataset = dataset
        self.denominator = denominator
 
    def heatmap(self, which):
        
        def ehr_domain(historic_ehr_df = self.historic_ehr_df):
            by_site1 = historic_ehr_df.copy()[['person_id','ehr_domain','ehr_site']].drop_duplicates()
            row_totals = by_site1[['person_id', 'ehr_domain']].drop_duplicates()
            row_totals = row_totals.groupby(['ehr_domain'], as_index = False).nunique()
            row_totals['ehr_site'] = 'TOTAL PIDS'

            by_site = pd.concat([by_site1.groupby(['ehr_site','ehr_domain'], as_index = False).nunique()
                                 , row_totals])
            by_site = by_site.pivot(index = ['ehr_site'] ,columns = ['ehr_domain']).fillna(0).astype('int')
            by_site.columns = [c[1] for c in by_site.columns]
            by_site.index.names = ['EHR Site']
            
            return by_site

        def vocabulary(historic_ehr_df = self.historic_ehr_df):
            ehr_vocab = historic_ehr_df[['person_id','ehr_site','vocabulary']].drop_duplicates() 

            remove_vocabs = ['Visit', 'CMS Place of Service', 'UB04 Pt dis status','Medicare Specialty', 'None', 'OMOP Extension']
            other_vocab = ['Cancer Modifier','None', 'CIEL', 'DRG'
                           'OMOP Extension', 'NDC', 'CVX', 'Multum', 'HemOnc', 'PPI', 'Nebraska Lexicon']#+remove_vocabs
            icd10 = ['ICD10CM','ICD10PCS']
            rxnorm = ['RxNorm','RxNorm Extension']

            col_totals = ehr_vocab[['person_id','ehr_site']].drop_duplicates()
            col_totals = col_totals.groupby('ehr_site').nunique()
            col_totals.columns = ['Any EHR']

            vocab_by_site = ehr_vocab.copy()[~ehr_vocab.vocabulary.isin(remove_vocabs)].drop_duplicates()
            vocab_by_site.loc[vocab_by_site.vocabulary.isin(other_vocab),'vocabulary'] = 'Others'
            vocab_by_site.loc[vocab_by_site.vocabulary.isin(icd10),'vocabulary'] = 'ICD10'
            vocab_by_site.loc[vocab_by_site.vocabulary.isin(rxnorm),'vocabulary'] = 'RxNorm'

            row_totals = vocab_by_site.copy()[['person_id', 'vocabulary']].drop_duplicates()
            row_totals = row_totals.groupby(['vocabulary'], as_index = False).nunique()
            row_totals = row_totals.T
            row_totals.columns = list(row_totals.loc['vocabulary'])
            row_totals = row_totals.drop('vocabulary')   
            row_totals.index = ['TOTAL PIDS']
            row_totals['Any EHR'] = historic_ehr_df.person_id.nunique()

            vocab_by_site = vocab_by_site.groupby(['ehr_site','vocabulary'], as_index = False).nunique()
            vocab_by_site = vocab_by_site.pivot(index = ['ehr_site'] ,columns = ['vocabulary']).fillna(0).astype('int')
            vocab_by_site.columns = [c[1] for c in vocab_by_site.columns]
            vocab_by_site.index.names = ['EHR Site']

            vocab_by_site = pd.concat([vocab_by_site, col_totals],1).sort_values('Any EHR', ascending = False)
            vocab_by_site = pd.concat([row_totals, vocab_by_site])
            vocab_by_site = vocab_by_site[sorted(vocab_by_site.columns)]
            
            return vocab_by_site
  
        def heatmap_code(raw_heatmap_df, which = which, total_col= 'Any EHR', w = 23,l = 30
                         , denominator = self.denominator):

            #format heatmap data   
            columns_dict = dict()
            heatmap_df = raw_heatmap_df.fillna(0).astype('int64')
            for n_col in heatmap_df.columns:

                heatmap_df[n_col+' %'] = heatmap_df[n_col]/denominator
                columns_dict[n_col] = n_col+' %'

            heatmap_df.index.name =  'EHR Sites'
            heatmap_df = combine_count_and_perc(heatmap_df, columns_dict, drop = False)

            heatmap0_df = heatmap_df.sort_values(total_col, ascending = False)
            heatmap_clean_df = heatmap0_df[[c for c in heatmap0_df.columns if '%' not in c and '_str' not in c and 'and' not in c]]
            heatmap_annot = heatmap0_df[[c for c in heatmap0_df.columns if 'and %' in c ]]
            
            plot_title = f'\nCount (and %) of Participants with EHR Data by EHR Site & {which.title()}\n'
            #######
            sns.set(rc={'figure.figsize':(w, l)})
            sns.set(font_scale = 1.5)
            g = sns.heatmap(heatmap_clean_df.sort_values(total_col, ascending = False)
                            , annot=heatmap_annot.values
                            , annot_kws={'size': 16}
                            , fmt=""
                            , cmap="Blues", linewidths=.5 , linecolor='black', robust = True
                           )

            plt.title(plot_title, fontsize=20)
            plt.tick_params(right=False, top=True, labelright=False, labeltop=True)
            plt.show()

            current_cdr = dataset.split('.')[1] 
            print(f'''\n*The denominator for all percentages is the Total Number of Participants in the current CDR ({current_cdr}), N = {'{:,}'.format(denominator)}''')
    
        if which.lower() == 'domain':
            raw_heatmap_df = ehr_domain()
            heatmap_code(raw_heatmap_df = raw_heatmap_df)

        else:
            if which.lower() == 'vocabulary':
                raw_heatmap_df = vocabulary()
                heatmap_code(raw_heatmap_df = raw_heatmap_df, w = 28, l = 32)
                print("'Others' Category includes: 'Cancer Modifier','None', 'CIEL', 'DRG','OMOP Extension', 'NDC', 'CVX', 'Multum', 'HemOnc', 'PPI', 'Nebraska Lexicon'")

