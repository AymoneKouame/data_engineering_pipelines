from google.cloud import bigquery
client = bigquery.Client()
import os
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime

# For Data Visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from venn import generate_petal_labels, draw_venn, generate_colors #venn
    
from utilities import utilities as u

import warnings
warnings.filterwarnings('ignore')

def new_datatypes_summary():

    p_ehr_query = f"""
        SELECT DISTINCT person_id
        FROM `measurement` 
        JOIN `measurement_ext` USING(measurement_id)
        WHERE LOWER(src_id) = 'participant mediated ehr'

        UNION DISTINCT
        SELECT DISTINCT person_id
        FROM `condition_occurrence` 
        JOIN `condition_occurrence_ext` USING(condition_occurrence_id)
        WHERE LOWER(src_id) = 'participant mediated ehr'

        UNION DISTINCT
        SELECT DISTINCT person_id
        FROM `device_exposure`
        JOIN `device_exposure_ext` USING(device_exposure_id)
        WHERE LOWER(src_id) = 'participant mediated ehr'

        UNION DISTINCT
        SELECT DISTINCT person_id
        FROM `drug_exposure`
        JOIN `drug_exposure_ext` USING(drug_exposure_id)
        WHERE LOWER(src_id) = 'participant mediated ehr'

        UNION DISTINCT
        SELECT DISTINCT person_id
        FROM `observation`
        JOIN `observation_ext` USING(observation_id)
        WHERE LOWER(src_id) = 'participant mediated ehr'

        UNION DISTINCT
        SELECT DISTINCT person_id
        FROM `procedure_occurrence`
        JOIN `procedure_occurrence_ext` USING(procedure_occurrence_id)
        WHERE LOWER(src_id) = 'participant mediated ehr'

        UNION DISTINCT
        SELECT DISTINCT person_id
        FROM `visit_occurrence`
        JOIN `visit_occurrence_ext` USING(visit_occurrence_id)
        WHERE LOWER(src_id) = 'participant mediated ehr'

    """

    summary_df = u().client_read_gbq(f"""

        SELECT DISTINCT COUNT(DISTINCT person_id) as N_Participants, 'Total in CDR' as Data
        FROM `person`
        
        UNION ALL

        SELECT COUNT(distinct person_id) as N_Participants, 'AIAN Race Ethnicity' as Data
        FROM `ds_survey` 
        WHERE question_concept_id = 1586140 AND answer = 'What Race Ethnicity: AIAN'

        UNION ALL

        SELECT COUNT(distinct person_id) as N_Participants, 'Disability or Life Functioning Survey ' as Data
        FROM `concept_ancestor` 
        JOIN `observation` on (descendant_concept_id=observation_concept_id)
        WHERE ancestor_concept_id in (705190) #Life Functioning Survey

        UNION ALL

        SELECT COUNT(distinct person_id) as N_Participants, 'Life Functioning Survey' as Data
        FROM `survey_conduct` 
        WHERE survey_source_concept_id = 705190 #'Life Functioning Survey'

        UNION ALL

        SELECT COUNT(DISTINCT person_id) as N_Participants, 'CareEvolution PPI' as Data
        FROM `survey_conduct`
        JOIN `survey_conduct_ext` USING(survey_conduct_id)
        WHERE LOWER(src_id) LIKE '%tpc' #subcontract CE

        UNION ALL

        SELECT count(distinct person_id) as N_Participants, 'Wear Consent' as Data
        FROM `wear_study` 
        WHERE resultsconsent_wear= 'Yes'

        UNION ALL

        SELECT count(distinct person_id) as N_Participants, 'Healthpro Deceased' as Data
        FROM `aou_death` 
        WHERE LOWER(src_id) LIKE '%healthpro%'

        UNION ALL

        SELECT count(distinct person_id) as N_Participants, 'Remote self-Reported Height PM' as Data
        FROM `measurement`
        JOIN `person_ext` USING(person_id)
        WHERE measurement_type_concept_id = 32865 and measurement_source_concept_id = 903133 #self-reported height #903121 ewight


        UNION ALL

        SELECT count(distinct person_id) as N_Participants, 'Remote self-Reported Weight PM' as Data
        FROM `measurement`
        JOIN `person_ext` USING(person_id)
        WHERE measurement_type_concept_id = 32865 and measurement_source_concept_id = 903121 #self-reported weight

        UNION ALL
        ----Participant Mediated EHR 
        SELECT count(distinct person_id) as N_Participants, 'Mediated EHR data' as Data
        FROM ({p_ehr_query})
        
        UNION ALL
        
        SELECT DISTINCT COUNT(DISTINCT person_id) as N_Participants, 'Self-Reported Racial/Ethnicity Subcategories' as Data
        FROM `ds_survey` 
        WHERE question_concept_id IN (1586150, 1586151, 1586152, 1586156, 1586153, 1586154, 1586149
                                , 1586139, 1586155, 1585599) #race specifics

        """)
    
    return summary_df.sort_values('N_Participants').reset_index(drop = True)


class overall_summary:    
    ## Class to get ubr flags for each person in the current dataset    
    def __init__(self):
        self.dataset = u().dataset
        self.bucket = u().bucket
        self.client_read_gbq = u().client_read_gbq
        
    def aian_pids(self):
        aian_pids = self.client_read_gbq(f'''SELECT DISTINCT person_id FROM `ds_survey`
                                WHERE question_concept_id = 1586140 AND answer = 'What Race Ethnicity: AIAN' ''')
        return aian_pids
    
    def wear_pids(self):
        wear_pids = self.client_read_gbq(f'''SELECT DISTINCT person_id 
                                        FROM `wear_study` WHERE resultsconsent_wear= 'Yes' ''')
        return wear_pids

    def data_types(self):
    
        # Query to get a list of all participants and their data types
        data_types_df = self.client_read_gbq(f"""
            SELECT 
                DISTINCT p.person_id
                , has_ppi_survey_data AS PPI             
                , has_physical_measurement_data AS PM 
                , has_ehr_data AS EHR
                , has_fitbit AS Fitbit                
                , has_whole_genome_variant AS WGS
                , has_array_data as Arr

            FROM `person` p
            LEFT JOIN `cb_search_person` USING(person_id) """)
        data_types_df = data_types_df.rename(columns = {'Arr':"Array"})
        data_types_df = data_types_df.fillna(0).astype('int64')

        return data_types_df
    
    def add_on_data_types(self):
    
        # Query to get a list of all participants and their data types
        data_types_df = self.client_read_gbq(f"""
            SELECT 
                DISTINCT p.person_id
                , has_lr_whole_genome_variant as Long_Read_WGS
                , has_structural_variant_data as Strutural_Variant
                , CASE WHEN has_whole_genome_variant = 1 OR has_array_data = 1 THEN 1 END AS WGS_OR_Array
                , CASE WHEN has_whole_genome_variant = 1 AND has_array_data = 1 THEN 1 END AS WGS_AND_Array

            FROM `person` p
            JOIN `cb_search_person` USING(person_id) """)
        data_types_df = data_types_df.fillna(0).astype('int64')

        return data_types_df
    
    def add_data_summaries(self, add_data_types_df, pop_dict):
    
        sum_report_add0 = pd.DataFrame()

        for pop, df_pop in pop_dict.items():
            if 'CDR' not in pop: add_data_types_df = add_data_types_df.merge(df_pop)
            cdr_add = self.count_by_data_type(add_data_types_df)
            cdr_add = cdr_add[cdr_add['Data Type']!= 'None']
            cdr_add.columns = ['Data Type', 'Count', 'Perc']
            cdr_add['Population'] = pop

            sum_report_add0 = pd.concat([sum_report_add0, cdr_add])#.set_index(['Population', 'Data Type'])
            
        sum_report_add0['Data Type'] = [i.replace('_',' ') for i in sum_report_add0['Data Type']]

        sum_report_add_f = self.combine_count_and_perc(sum_report_add0, count_perc_cols_dic = {'Count':'Perc'})
        sum_report_add = sum_report_add_f.pivot(index = ['Data Type'], columns =['Population']).reset_index()
        sum_report_add.columns = [c[1] for c in sum_report_add.columns]
        sum_report_add = sum_report_add.rename(columns = {'':'Data Type'})

        return sum_report_add.fillna('-')

    def calc_percentage(self, df, denom):
        df['% of total participants in CDR'] = df['Count of Participants in CDR']/denom
        df['% of total participants in CDR'] = ['{:.2%}'.format(i) for i in df['% of total participants in CDR']]

        return df

    def count_by_data_type(self, df):
        #display(data_types_df)
        denom = df.person_id.nunique()
        
        DF = df.melt(id_vars = ['person_id'])
        DF = DF[DF.value == 1].drop('value', axis =1)

        DF_none = df.loc[df[df.drop('person_id', axis =1).columns].sum(axis = 1) == 0][['person_id']]
        DF_none['variable'] = 'None'

        DF = pd.concat([DF, DF_none])

        DF_count = DF.groupby(['variable'], as_index = False).nunique().sort_values('person_id')
        DF_count.columns = ['Data Type','Count of Participants in CDR']

        DF_count = self.calc_percentage(DF_count, denom = denom)\
            .rename(columns = {'Count of Participants in CDR':'Count of Participants in CDR ('+'{:,}'.format(denom)+')'})

        return DF_count


    def count_by_n_data_type(self, df):
        
        denom = df.person_id.nunique()
        n_data_types_df = pd.DataFrame(df.set_index('person_id').sum(axis = 1)
                                      , columns = ['n_data_types']).reset_index()


        ## By number of data type
        DF_count = n_data_types_df.groupby(['n_data_types'], as_index = False).nunique()
        DF_count.columns = ['Number of Data Types', 'Count of Participants in CDR']

        max_n = int(DF_count['Number of Data Types'].max())
        DF_count['Number of Data Types'] = ['Participants with \nONLY '+str(int(i))+' Data Type(s)' \
                                            for i in DF_count['Number of Data Types']]   
        DF_count = pd.concat([DF_count
                              , pd.DataFrame({'Number of Data Types': 'TOTAL PARTICIPANTS IN CDR'
                                              , 'Count of Participants in CDR':DF_count['Count of Participants in CDR'].sum()}
                                             , index = [''])])

        DF_count = self.calc_percentage(DF_count, denom = denom)
        DF_count['Number of Data Types'] = DF_count['Number of Data Types']\
                                                .replace({'Participants with \nONLY 0 Data Type(s)':'Participants with \n0 Data Type(s)'
                                                 , 'Participants with \nONLY '+str(max_n)+' Data Type(s)':'Participants with \nall '\
                                                  +str(max_n)+' Data Type(s)'
                                                     })

        ## By number and type of Data
        DF_none = n_data_types_df[n_data_types_df.n_data_types ==0]
        DF_none['data_types'] = 'None'

        DF = df.melt(id_vars = ['person_id'])
        DF = DF[DF.value ==1].drop('value', axis =1)
        DF = DF.merge(n_data_types_df[n_data_types_df.n_data_types >0])

        DF['data_types'] = DF.groupby(['person_id','n_data_types'])['variable'].transform(lambda x: ' and '.join(x))
        DF = DF[['person_id','data_types']].merge(n_data_types_df[n_data_types_df.n_data_types >0])

        DF_count_d = pd.concat([DF, DF_none])
        DF_count_d = DF_count_d.groupby(['n_data_types','data_types'], as_index = False).nunique()
        DF_count_d.columns = ['Number of Data Types','Data Types', 'Count of Participants in CDR']


        DF_count_d['Number of Data Types'] = ['Participants with ONLY '+str(int(i))+' Data Type(s)' for i in DF_count_d['Number of Data Types']]

        DF_count_d = pd.concat([DF_count_d
                                , pd.DataFrame({
                                               'Number of Data Types': ' '
                                               , 'Count of Participants in CDR':DF_count_d['Count of Participants in CDR'].sum()
                                               , 'Data Types': 'TOTAL PARTICIPANTS IN CDR'
                                               }, index = [''])])

        DF_count_d = self.calc_percentage(DF_count_d, denom = denom)

        DF_count_d['Number of Data Types'] = DF_count_d['Number of Data Types']\
                        .replace({'Participants with ONLY 0 Data Type(s)':'Participants with 0 Data Types'
                                 , 'Participants with ONLY '+str(max_n)+' Data Type(s)':'Participants with all '\
                                  +str(max_n)+' Data Type(s)'
                                 })
        DF_count_d['Count of Participants in CDR'] = DF_count_d['Count of Participants in CDR'].astype('int')

        return DF_count, DF_count_d

    def format_numbers(self, df):
        # format the counts and percentage columns

        formated_df = df.copy()
        for col in formated_df.columns:
            if formated_df[col].dtype == 'int64':
                formated_df[col] = ['{:,}'.format(i) for i in formated_df[col]]

        return formated_df

    def combine_count_and_perc(self, df, count_perc_cols_dic, drop = True):    
        df_comb = df.copy()

        for count_col in count_perc_cols_dic:
            perc_col = count_perc_cols_dic[count_col]
            df_comb[count_col+'_str'] = df_comb[count_col]
            df_comb[perc_col+'_str'] = df_comb[perc_col]
            df_comb[[count_col+'_str', perc_col+'_str']] = self.format_numbers(df_comb[[count_col+'_str', perc_col+'_str']])
            new_col_name = count_col+' (and %)'
            df_comb[new_col_name] = df_comb[count_col+'_str'].astype('str')+' ('+df_comb[perc_col+'_str'].astype('str')+')'

            if drop == True:
                df_comb = df_comb.drop([count_col, count_col+'_str', perc_col, perc_col+'_str'], axis =1)
        return df_comb

    def format_dataframe(self, df):

        display_DF = pd.DataFrame()
        for n in df['Number of Data Types'].unique():
            DF = df[df['Number of Data Types'] == n].drop('Number of Data Types', axis =1)
            DF = DF.reindex([n]+ DF.index.tolist()).fillna('')
            display_DF = pd.concat([display_DF, DF])

        index_list = display_DF.index.tolist()
        index_list = [i for i in index_list if type(i) == int]

        dic = {0:''}
        for i in index_list:
            dic.update({i:''})
        display_DF = display_DF.rename(index=dic)
        return display_DF


    def combine_final_tables(self, dfs_to_combine, indexname, save = False):
        table = pd.DataFrame()
        for colname, df in dfs_to_combine.items():
            DF = df.set_index(indexname)
            DF.columns = [colname]
            table = pd.concat([table,DF],axis =1).fillna('-')
            
        ###################### save the file to the bucket
        if save == True:
            add = indexname[0].lower().replace(' ', '_')
            filename = f'1_count_by_{add}.xlsx'
            u().write_to_csv_excel(table, filename = filename)
 
        return table
    
    def table_1_by_pop(self, pop_df):
        n = pop_df.person_id.nunique()
        table1_1_uncomb = self.count_by_data_type(df = pop_df)
        table1_2_uncomb, table1_2_raw = self.count_by_n_data_type(df = pop_df)

        count_col = "Count of Participants in CDR"; perc_col = "% of total participants in CDR"
        table1_1 = self.combine_count_and_perc(table1_1_uncomb
                                        , count_perc_cols_dic= dict({f"{count_col} ({'{:,}'.format(n)})":f'{perc_col}'}))
        table1_2 = self.combine_count_and_perc(table1_2_raw, count_perc_cols_dic= dict({f'{count_col}':f'{perc_col}'}))
        table1_2 = table1_2.reset_index(drop = True)

        return table1_1, table1_2, table1_1_uncomb, table1_2_uncomb

#     def venn_plot(self, data_types_df, pid_type, exclude = None, save = False):

#         all_datatypes = ['PPI', 'EHR', 'PM', 'Fitbit']
#         data = {'Genomics': set(data_types_df[(data_types_df.Array == 1) | (data_types_df.WGS ==1)].person_id)}
#         for dtype in all_datatypes: data.update({dtype: set(data_types_df[data_types_df[dtype] == 1].person_id)})
#         data['Phys. Meas.'] = data.pop('PM')
        
#         if exclude != None:
#             #print(f'Excluding {exclude}.')
#             for i in exclude: del data[i]

#         v = venn(data)
#         ptype = pid_type.replace('AIAN', 'AI/AN')
#         plt.title(f'Count of {ptype} participants with multiple data types \n'\
#                   .replace(f'AIAN participants', 'participants who self-identify as AI/AN')
#                   , fontsize = 18)
#         plt.tight_layout()
#         if save == True: 
#             add = ','.join(list(data.keys()))
#             filename = f'venn_{pid_type}_{add}.jpeg'
#             plt.savefig(filename)      
#             args = ["gsutil", "cp", f"./{filename}", f"{self.bucket}/notebooks/data"]
#             output = subprocess.run(args, capture_output=True)


    def venn_plot(self, data_types_df, pid_type, w=16, l=8, exclude = None, save = False):

        all_datatypes = ['PPI', 'EHR', 'PM', 'Fitbit']
        data = {'Genomics': set(data_types_df[(data_types_df.Array == 1) | (data_types_df.WGS ==1)].person_id)}
        for dtype in all_datatypes: data.update({dtype: set(data_types_df[data_types_df[dtype] == 1].person_id)})
        data['Phys. Meas.'] = data.pop('PM')

        if exclude != None:
            for i in exclude: del data[i]

        petal_labels = generate_petal_labels(data.values(), fmt="{size}")
        for k,v in petal_labels.items():
            if int(v) <20: v = '<20'
            else: v = "{:,}".format(int(v))    
            petal_labels[k] = v

        draw_venn(
            petal_labels=petal_labels, dataset_labels=data.keys(),
            hint_hidden=False, colors=generate_colors(n_colors=len(data.keys())),
            figsize=(w, l), fontsize=12, legend_loc="best", ax=None
        )

        plt.title(f'Count of {pid_type} participants with multiple data types \n'\
                        .replace(f'AIAN participants', 'participants who self-identify as AI/AN')
                        , fontsize = 18)
        plt.tight_layout()
        if save == True: 
            add = ','.join(list(data.keys()))
            filename = f'venn_{pid_type}_{add}.jpeg'
            plt.savefig(filename)      
            args = ["gsutil", "cp", f"./{filename}", f"{self.bucket}/notebooks/data"]
            output = subprocess.run(args, capture_output=True)
            
            
    def combine_plot_data(self, xcol, dfs_to_combine):

        count_col = "Count of Participants in CDR"
        perc_col = "% of total participants in CDR" 

        ## whole cdr
        kw = 'Whole CDR'
        df_cdr_raw = dfs_to_combine[kw]
        df_cdr = df_cdr_raw.copy()
        df_cdr['Participant'] = kw
        df_cdr = df_cdr.rename(columns = {f"{[c for c in df_cdr.columns if 'Count of Participants' in c][0]}":f'{count_col}'})


        DF = pd.DataFrame()
        for key in [k for k in dfs_to_combine.keys() if k != 'Whole CDR']:
            df = dfs_to_combine[key]
            n = df.iloc[-1,1]
            df['Number of Data Types'] = df[xcol]#.replace('Participants with \nall 5 Data Type(s)','Participants with \nONLY 5 Data Type(s)')
            rename_c = [c for c in df.columns if f'{count_col}' in c][0]
            df = df.rename(columns = {rename_c:f'{count_col}'})

            missing_rows = set(df_cdr[xcol]) - set(df[xcol])
            none = pd.DataFrame(missing_rows, columns = [xcol])
            none[f'{count_col}'] = 0
            none[f'{perc_col}'] = '0%'
            none[xcol] = 'Participants with \n0 Data Type(s)'

            df = pd.concat([none, df]).reset_index(drop = True)
            df['Participant'] = key
            DF = pd.concat([DF, df]).drop_duplicates()

        DF = pd.concat([df_cdr, DF]).drop_duplicates()       
        return DF

    def create_plot_label(self, df_plot, count_col ='Count of Participants in CDR'
                         , perc_col = '% of total participants in CDR'):
        df_plot_final = df_plot.drop_duplicates().reset_index(drop = True)

        df_plot_final[perc_col] = [round(float(c.replace('%', ''))) for c in df_plot_final[perc_col]]
        df_plot_final['label'] = ['{:,}'.format(i) for i in df_plot_final[count_col]] 
        df_plot_final['label'] = df_plot_final['label']+ ' ('+ df_plot_final[perc_col].astype('str')+ '%)'
        df_plot_final['label'] = [c.replace(' (', '\n(') for c in df_plot_final['label']]
        return df_plot_final

    def bar_plot(self, df, title, label, y, x= 'Data Type', hue= 'Participant'
                     , w = 14, h = 10, r = 0, ha = 'left', va = 'bottom', save = False):

        plt.figure(figsize=(w,h), tight_layout=True)
        ax = sns.barplot(x= x, y = y, hue = hue, palette='Blues', ci=None, edgecolor = 'w', data = df)

        for p in ax.patches:
            n = int(p.get_height())
            l = df.loc[df[y] == n, label].reset_index(drop = True)[0]

            t = ax.annotate(l, xy = (p.get_x(), n))
            t.set(color = "black", size = 12, rotation=r, horizontalalignment= ha, verticalalignment=va)

        ax.set(xlabel="",ylabel="")
        ax.set_title(title+'\n\n', fontsize=18)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=r, horizontalalignment= 'center', fontsize=12)
        ax.set(yticklabels=[])
        plt.show()
        
        if save == True:
            filename = title.replace('Count of Participants in the CDR ', 'Barplot ')+'.jpeg'
            plt.savefig(filename)        
            args = ["gsutil", "cp", f"./{filename}", f"{self.bucket}/notebooks/data"]
            output = subprocess.run(args, capture_output=True)
            print(output.stderr)




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


