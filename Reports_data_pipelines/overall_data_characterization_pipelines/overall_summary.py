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