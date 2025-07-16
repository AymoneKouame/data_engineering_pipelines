
#1
from new_datatypes_summary import *
summary_df = new_datatypes_summary()

#2
from overall_summary import *

s = overall_summary()
data_types_df = s.data_types()
dfs_to_combine = {'Whole CDR': sum_reports_cdr[0], "WEAR":sum_reports_wear[0]}

sum_reports_cdr = s.table_1_by_pop(data_types_df)
counts_by_datatype = s.combine_final_tables(dfs_to_combine, indexname = ['Data Type'], save = False)
bar_plot_data1 = s.combine_plot_data(dfs_to_combine = dfs_to_combine, xcol='Data Type').drop('Number of Data Types', axis = 1)
bar_plot_data1 = s.create_plot_label(bar_plot_data1)

s.bar_plot(bar_plot_data1
         , x= 'Data Type', y = 'Count of Participants in CDR'
         , label = 'label'
         , title = 'Count of Participants in the CDR by Data Types', w = 16)


#3
from demographics_by_datatypes import *
dbd = demographics_by_datatypes()
table2_1 = dbd.combine_dataframes('race', demographics_df, dbd.race_index_order, all_cdr_pids)
u().p_display(table2_1)

#4
from genomics_by_phenotypes import *
gen = genomics_by_phenotypes()
genomic_data = gen.data_types()