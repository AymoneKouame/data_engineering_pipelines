# Functions and code to collect, wrangle and report on historical EHR data growth
# Set Up

import pandas as pd
import numpy as np
import statistics
import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import os

dataset = %env WORKSPACE_CDR

def pid_cumcount_by_vars(df, cum_count_by):
    cummulative_counts_df = pd.DataFrame()
    for d in df[cum_count_by[0]].unique():
        print('\n'+d)
        DF = df[df[cum_count_by[0]] == d][['person_id']+cum_count_by]\
                .sort_values(cum_count_by, ascending = True)
        DF['cumcount'] = (~DF['person_id'].duplicated()).cumsum()
        cum_df = DF.drop('person_id',1).drop_duplicates().reset_index(drop = True)
        cum_df[cum_count_by[0]] = d
        cum_df = cum_df.sort_values(cum_count_by)\
                .groupby(cum_count_by, as_index = False).last() 
        
        
        print('  QC: N pids, min cumcount, max cumcount: '+\
              str(DF["person_id"].nunique()), str(cum_df['cumcount'].min()), str(cum_df['cumcount'].max()))
        
        cummulative_counts_df = pd.concat([cummulative_counts_df, cum_df])
        
    cummulative_counts_df['cumcount'] = cummulative_counts_df['cumcount'].astype('int')

    return cummulative_counts_df

def transform_and_plot(df, denominator = denominator, add_perc_text = 'yes', rotate_x = 0, save_plot = 'yes'):
    
    #Transform
    df_plot = df.drop(['cumcount'],1)\
                    .pivot(index = ['year'], columns = ['ehr_domain']).fillna(0)#.reset_index()
    df_plot.columns = [c[1] for c in df_plot.columns]
    
    current_cdr = dataset.split('prod.')[1]
    fig_caption = f'''This plot represents EHR data availability in the current CDR ({current_cdr}). 
    The denominator for the percentages is the Total Number of Participants in the current CDR, N = {'{:,}'.format(denominator)}.
    For better visibility, the plot only displays EHR data growth starting in 2010. '''
    
    #plot
    plt.figure(figsize=(16,8), tight_layout=True)
    plt.plot(df_plot, 'o-', linewidth=2)
    
    ##customization
    #plt.xticks([2020, 2021, 2022])
    plt.xticks(df_plot.index, size = 13, rotation = rotate_x)
    plt.yticks(range(int(df['cum%'].min()), int(df['cum%'].max()), 5),size = 13)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(decimals=0))
    plt.xlabel('Year', size = 15)
    plt.ylabel('% of Participants with EHR data', size = 15)
    plt.title('Historical Availability of EHR Records in Current CDR', size = 16)

    plt.legend(title='EHR Domains', title_fontsize = 14, labels=df_plot.columns, fontsize = 14)
    plt.text(statistics.median(test.index), -16
         , fig_caption, verticalalignment='bottom',style='italic'
         , horizontalalignment='center', color = 'black', size = 12)

    
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

        for d in ['Condition', 'Device Exposure', 'Drug Exposure'#,'Measurement'
                  ,'Observation'#,'Procedure'
                  ,'Visit']:
            plt.text(min_year-0.1, df_plot[d][min_year]-0.6
                     , str(int(df_plot[d][min_year]))+'%',verticalalignment='center'
                     ,horizontalalignment='right', color = 'black', size = 12, rotation = 13)

        for d in ['Condition', 'Device Exposure', 'Drug Exposure'#,'Measurement'
                  ,'Observation'#,'Procedure','Visit'
                 ]:

            plt.text(max_year+0.2, df_plot[d][max_year]-1
                     , str(int(df_plot[d][max_year]))+'%',verticalalignment='bottom'
                     ,horizontalalignment='center', color = 'black', size = 12)

    if save_plot.lower() == 'yes':
        plt.savefig('HistoricEHR_'+str(current_cdr)+"_"+str(datetime.datetime.now().date())+'.jpeg')
    plt.show()

# Collect Historic EHR data
print('Measurement', datetime.datetime.now())
historic_meas = pd.read_gbq(f'''
            SELECT
               DISTINCT person_id, 'Measurement' as ehr_domain, MIN(EXTRACT(YEAR FROM measurement_date)) as year
            FROM `{dataset}.measurement` AS m
            LEFT JOIN `{dataset}.measurement_ext` AS mm ON m.measurement_id = mm.measurement_id
            WHERE LOWER(mm.src_id) LIKE 'ehr site%'
            GROUP BY 1,2
            ''')

print('Condition', datetime.datetime.now())
historic_cond = pd.read_gbq(f'''

            SELECT
               DISTINCT person_id, 'Condition' as ehr_domain
               , MIN(EXTRACT(YEAR FROM condition_start_date)) as year
            FROM `{dataset}.condition_occurrence` AS m
            LEFT JOIN `{dataset}.condition_occurrence_ext` AS mm ON m.condition_occurrence_id = mm.condition_occurrence_id
            WHERE LOWER(mm.src_id) LIKE 'ehr site%'
            GROUP BY 1,2
            ''')

print('Device', datetime.datetime.now())            
historic_device = pd.read_gbq(f'''

            SELECT
               DISTINCT person_id, 'Device Exposure' as ehr_domain
               , MIN(EXTRACT(YEAR FROM device_exposure_start_date)) as year
            FROM `{dataset}.device_exposure` AS m
            LEFT JOIN `{dataset}.device_exposure_ext` AS mm ON m.device_exposure_id = mm.device_exposure_id
            WHERE LOWER(mm.src_id) LIKE 'ehr site%'
            GROUP BY 1,2
            ''')

print('Drug', datetime.datetime.now())
historic_drug = pd.read_gbq(f'''

            SELECT
               DISTINCT person_id, 'Drug Exposure' as ehr_domain
               , MIN(EXTRACT(YEAR FROM drug_exposure_start_date)) as year
            FROM `{dataset}.drug_exposure` AS m
            LEFT JOIN `{dataset}.drug_exposure_ext` AS mm ON m.drug_exposure_id = mm.drug_exposure_id
            WHERE LOWER(mm.src_id) LIKE 'ehr site%'
            GROUP BY 1,2
            ''')
            
print('Observation', datetime.datetime.now())
historic_obs = pd.read_gbq(f'''

            SELECT
               DISTINCT person_id, 'Observation' as ehr_domain
               , MIN(EXTRACT(YEAR FROM observation_date)) as year
            FROM `{dataset}.observation` AS m
            LEFT JOIN `{dataset}.observation_ext` AS mm ON m.observation_id = mm.observation_id
            WHERE LOWER(mm.src_id) LIKE 'ehr site%'
            GROUP BY 1,2
            ''')

print('Procedure', datetime.datetime.now())
historic_proc = pd.read_gbq(f'''

            SELECT
               DISTINCT person_id, 'Procedure' as ehr_domain
               , MIN(EXTRACT(YEAR FROM procedure_date)) as year
            FROM `{dataset}.procedure_occurrence` AS m
            LEFT JOIN `{dataset}.procedure_occurrence_ext` AS mm ON m.procedure_occurrence_id = mm.procedure_occurrence_id
            WHERE LOWER(mm.src_id) LIKE 'ehr site%'
            GROUP BY 1,2 ''')

print('Visit', datetime.datetime.now())
historic_visit = pd.read_gbq(f'''

            SELECT
               DISTINCT person_id, 'Visit' as ehr_domain
               , MIN(EXTRACT(YEAR FROM visit_start_date)) as year
            FROM `{dataset}.visit_occurrence` AS m
            LEFT JOIN `{dataset}.visit_occurrence_ext` AS mm ON m.visit_occurrence_id = mm.visit_occurrence_id
            WHERE LOWER(mm.src_id) LIKE 'ehr site%' 
            GROUP BY 1,2 ''')


print('Overall', datetime.datetime.now())
historic_overall = pd.read_gbq(f'''

            SELECT
               DISTINCT person_id, 'Overall' as ehr_domain
               , MIN(EXTRACT(YEAR FROM visit_start_date)) as year
            FROM `{dataset}.visit_occurrence` AS m
            LEFT JOIN `{dataset}.visit_occurrence_ext` AS mm ON m.visit_occurrence_id = mm.visit_occurrence_id
            WHERE LOWER(mm.src_id) LIKE 'ehr site%' 
            GROUP BY 1,2 ''')

historic_ehr_raw = pd.concat([historic_meas, historic_cond
                          , historic_device, historic_drug
                          , historic_obs, historic_visit, historic_proc])


# Run Report
historic_overall = historic_ehr_raw.copy()
historic_overall['ehr_domain'] = 'Any EHR'
historic_overall = historic_overall.drop_duplicates()#.groupby(['person_id','ehr_domain'], as_index= False).min()
historic_ehr = pd.concat([historic_ehr_raw, historic_overall])
historic_ehr.person_id.nunique()

## Cummulative counts
historic_ehr_cum = pid_cumcount_by_vars(df  = historic_ehr.copy(), cum_count_by = ['ehr_domain','year'])
historic_ehr_cum.head()

## Plot
cut_off_date = 2010
historic_ehr_cum_after2000 = historic_ehr_cum[historic_ehr_cum.year >= cut_off_date]
historic_ehr_cum_before2000 = historic_ehr_cum[historic_ehr_cum.year < cut_off_date]
historic_ehr_cum_1900to2000 = historic_ehr_cum[historic_ehr_cum.year.between(1900,2000)]

transform_and_plot(df = historic_ehr_cum_before2000, add_perc_text='no', save_plot = 'no', rotate_x = 90)