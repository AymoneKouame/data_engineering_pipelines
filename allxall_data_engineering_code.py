%%writefile utilities.py

import pandas as pd
import os
import subprocess
from google.cloud import bigquery
import nbformat as nbf
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor
import shutil
import time
from datetime import datetime

client = bigquery.Client()

def client_read_gbq(query, dataset = os.getenv('WORKSPACE_CDR')):
    
    job_config = bigquery.QueryJobConfig(default_dataset=dataset)
    query_job = client.query(query, job_config =job_config)  # API request
    df = query_job.result().to_dataframe()

    return df

# def read_from_bucket(filename, n = None, directory = 'notebooks/phenotype_data'):   
#     my_bucket = os.getenv('WORKSPACE_BUCKET')
#     ext = filename.split('.', -1)[-1]
    
#     args = ["gsutil", "cp", f'{my_bucket}/{directory}/{filename}', './']
#     output = subprocess.run(args, capture_output=True)
#     print(output.stderr)    
    
#     if ext == 'csv': df = pd.read_csv(filename, nrows = n)
#     elif ext == 'xlsx': df = pd.read_excel(filename, nrows = n)
#     elif ext =='tsv': df = pd.read_csv(filename, sep='\t', nrows = n)
#     else: print('ERROR: file must be excel, tsv or csv.')
        
#     return df

# def save_to_bucket(df, filename, directory = 'notebooks/phenotype_data', save_index = False):
#     my_bucket = os.getenv('WORKSPACE_BUCKET')
#     ext = filename.split('.', -1)[-1]
    
#     if ext == 'csv': df.to_csv(filename, index = save_index)
#     elif ext == 'xlsx': df.to_csv(filename, index = save_index)
#     elif ext =='tsv': df.to_csv(filename, sep='\t', index = save_index)
#     else: print('ERROR: file must be excel, tsv or csv.')

#     args = ["gsutil", "cp", f'{filename}', f'{my_bucket}/{directory}/']
#     output = subprocess.run(args, capture_output=True)
#     print(output.stderr)

def make_clickable(url, name):
    return f'<a href="{url}">{name}</a>'

def click(df):
    DF = df.copy()
    DF['notebook_url'] = DF.apply(lambda x: make_clickable(x['notebook_url'], x['notebook_name']), axis=1)
    DF = DF.drop('notebook_name', axis= 1)
    return DF.style.format({'url': make_clickable})

def create_phenotype_index_file(phe_filename, col_name, dtype, save = True):
    
    # create phenotype index for all datatypes, except lab
    df = pd.read_csv(f'{phe_filename}', nrows = 0)
    concepts = df.drop('person_id', axis = 1).columns
    if col_name == 'concept_id':concepts_query = tuple([int(i) for i in concepts])
    else: concepts_query = tuple(concepts)
        
    if 'physical_measurement' in phe_filename:
        col_name
        query = f"""SELECT DISTINCT concept_id, LOWER(concept_name) as concept_name, LOWER({col_name}) as {col_name}
                    FROM `measurement` JOIN `concept` on measurement_concept_id = concept_id
                    WHERE LOWER({col_name}) IN ('weight', 'height', 'waist-circumference-mean', 'hip-circumference-mean'
                                                , 'heart-rate-mean', 'blood-pressure-diastolic-mean'
                                                , 'blood-pressure-systolic-mean') """
    else:
        query = f'''SELECT DISTINCT {col_name}, LOWER(concept_name) as concept_name 
                    FROM `concept` WHERE {col_name} IN {concepts_query} '''
        
    phe_index = client_read_gbq(query).sort_values('concept_name')
    
    if 'pfhh_survey' in phe_filename:
        q = 'including yourself, who in your family has had'
        phe_index['short_name'] = [i.replace(q, '').replace('select all that apply.', '') for i in phe_index['concept_name']]
        phe_index = phe_index.sort_values(['short_name'])
    
    ##
    pwd = os.getenv('PWD')
    nspace = os.getenv('WORKSPACE_NAMESPACE')
    wk_id = nspace+'/'+pwd.split('workspaces/')[1]
    workspace_link = f'https://workbench.researchallofus.org/workspaces/{wk_id}/analysis/preview'

    phe_index['notebook_name'] = [get_notebook_name(dtype, i) for i in phe_index[col_name]]
    phe_index['notebook_url'] = workspace_link+'/'+phe_index['notebook_name']
    
    if save == True: 
        BUCKET = os.getenv('WORKSPACE_BUCKET')
        phe_index.to_csv(f'{BUCKET}/notebooks/phenotype_data/{phe_filename}'.replace('table','index'))
        #save_to_bucket(phe_index, phe_filename.replace('table','index'), directory = 'notebooks/phenotype_data')
    
    return phe_index

def add_link_lab_index(lab_index_filename, save = True):
    
    ## Lab index already exisist. Add column with notebook link
    pwd = os.getenv('PWD')
    nspace = os.getenv('WORKSPACE_NAMESPACE')
    wk_id = nspace+'/'+pwd.split('workspaces/')[1]
    workspace_link = f'https://workbench.researchallofus.org/workspaces/{wk_id}/analysis/preview'
    
    phe_index = pd.read_csv(lab_index_filename)
    phe_index['notebook_name'] = 'measurement_'+phe_index['measurement_concept_id']+'.ipynb'
    phe_index['notebook_url'] = workspace_link+'/'+phe_index['notebook_name']
    
    if save == True: 
        BUCKET = os.getenv('WORKSPACE_BUCKET')
        phe_index.to_csv(f'{BUCKET}/notebooks/phenotype_data/{lab_index_filename}')
        #save_to_bucket(phe_index, lab_index_filename, directory = 'notebooks/phenotype_data')
    
    return phe_index

def get_notebook_code(datatype, concept_id, phe_filename):
    
############################
    if 'lab' in datatype.replace('_',' '):
        mrkd_intro = f"""
<div class='alert alert-info' style='font-family:Arial; font-size:20px' ><center><font size="6"> Lab Measurement {concept_id} </font></center></div>

This notebook details how the laboratory measurement {concept_id} was curated for All by All analysis. The sections are as follows:

- **Notebook setup**: Import Unifier python library, read reference tables into memory.
- **Unit harmonization and outlier removal**: Select measurement of interest, harmonize all reported units to a standard unit, and remove outlier values.
- **Data visualization**: Plot the data before and after unit harmonization to assess efficacy and identify the need for further quality control processes.
- **Quality control**: Drop data for the selected laboratory measurement at sites with data quality concerns after unit harmonization, as needed.
- **Changing outlier boundaries**: Change the thresholds used to define outliers for the selected laboratory measurement, as needed.

**Note:**
- If you have not already, **run the set up in the '0_ReadMe' once** before running the notebooks in this workspace.
- This notebook has been automatically generated. If you wish to rerun it, you may need to choose the Python kernel (top menu).
    
# Notebook setup
In this section you will import the Unifier python library and read reference tables into memory. """   
        
        code1 = f"""from unifier_functions import *
from IPython.core.interactiveshell import InteractiveShell

# Read in tables for unit processing
tables = read_tables(pipeline_version='v1_0')
metadata = tables['metadata']
unit_map_table = tables['unit_map']
unit_reduce_table = tables['unit_reduce']"""
        
        mrkd1 = f"""# Unit harmonization and outlier removal
In this section you will perform unit harmonization for measurement(s) of interest and remove outlier values.

## Select measurement of interest
The cell below is used to select the measurement of interest by defining the measurement_concept_id.

Please note that multiple measurement_concept_id's can be assigned to process the data for these measurements as seen in Example 2."""
        
        code2 = f"""# Define m_cid as a list of integers containing the OMOP measurement_concept_id for the measurement of interest
m_cid = [3023314]

# Query CDR v7 Controlled Tier
measurement = measurement_data(m_cid)

# Summarize OMOP query
query_summary(measurement)"""      
        
        mrkd2 = f"""# Harmonize units and drop outlier values
The cell below will harmonize all units to a standard unit as defined in the resource *metadata* by applying conversion factors specified in the resource *unit_reduce_table. Outlier values are dropped based on upper and lower thresholds in the metadata* resource.

The lifecycle of the selected measurement unit and value data is detailed in the *df_harmonized* dataframe.

The processed data is written as the *df_final* dataframe."""

        code3 = f"""# Preprocess the input measurement dataframe and ensure it meets the proper format for Unifier
preprocessed = preprocess(measurement)

# Produce dataframe with lifecycle of measurement unit harmonization and labeling missing/outlier values
df_harmonized = harmonize(preprocessed, metadata, unit_map_table, unit_reduce_table)

# Final dataframe with measurement data after completing unit harmonization and dropping outliers
df_final = trim(df_harmonized)

# Descriptive statistics after outliers are removed
unitdata = units_dist(df_harmonized)"""        
        
        mrkd3 = """The cell below will plot all data for the selected *measurement_concept_id* in the original reported units. The vertical red line indicates the upper threshold used to define outliers but is only applicable on the plot for values in the standard unit.

The standard unit and outlier thresholds for each measurement included in All by All are defined in the *metadata* table."""
        
        code4 = """InteractiveShell.ast_node_interactivity = "all"

max_value = metadata[metadata['measurement_concept_id']==m_cid[0]]['maximum_value'].iloc[0]
unit_list = df_harmonized['assigned_unit'].unique()
sorted_unit_list = np.sort(unit_list)
value_type = 'unedited_value_as_number'

print("You can change bin size using the 'bin_factor' argument.")
for unit_name in sorted_unit_list:
    plot_hist_single_lab_unit(df_harmonized, df_harmonized['lab_name'].iloc[0], unit_name, value_type,
                              maximum_value=max_value,low_cutoff=0, high_cutoff=max_value*10, bin_factor=100)"""    
        
        mrkd4 = """# Data visualization
In this section you will plot measurement data before and after unit harmonization and outlier filtering. The data will be formatted as boxplots and stratified by electronic health record site and unit type. This will help visualize the data to assess the effectiveness of the Unifier workflow on preparing measurement data in a research-ready format.

## Visualization prior to unit harmonization and outlier filtering
The cell below plots the original data in their reported units without processing. These plots do exclude frequencies of missing values and missing value equivalents (i.e. value_as_number == 10,000,000 AND value_as_number == 100,000,000).
"""        
        code5 = """#bplot_filtered displays all values for the lab measurement in the original units
bplot_filtered = boxplot(df_harmonized, 'annotated', fill_col='assigned_unit', value_col='unedited_value_as_number')
bplot_filtered"""
        
        mrkd5 = """## Visualization after data processing
The cell below plots the data which were able to be harmonized to the standard unit and were not labeled as outliers."""
        
        code6 = """# bplot_final displays only non-outlier values which were able to be harmonized to the standard unit
bplot_final = boxplot(df_final, 'final', fill_col='unit', value_col='value_as_number')
bplot_final"""
        
        mrkd6 = """# Quality Control
This section of the notebook allows the user to drop the data from a site which did not harmonize correctly. In the first line of code below, specfic site(s) can be inputted to drop all data corresponding to the site. The dataframe containing processed data is updated and are re-plotted to allow visualization of the data following removal of specific site(s) data. Note that multiple sites can be inputted in the first line of code as a list to drop values corresponding to all sites listed.
"""
        
        code7 = """#Drop all data from sites which did not harmonize correctly
df_harmonized_dropped = drop_ehr_site(df_harmonized, which_sites = [537])

#Regenerate df_trimmed using the updated df_harmonized
df_trimmed_dropped = trim(df_harmonized_dropped)

#Plot bplot_final again after removing sites which did not harmonize correctly
bplot_final = boxplot(df_trimmed_dropped, 'final', fill_col='unit', value_col='value_as_number')
bplot_final"""
        
        mrkd7 = """Changing outlier boundaries
This section of code allows users to update the upper and lower outlier thresholds as needed. Sections 1 - 3 of the notebook should be re-run using the updated_metadata table after updating the bounds to curate the lab measurements using the newly defined bounds.

This cell outputs the original minimum and maximum value for the lab measurement defined in the metadata table."""
      
        code8 = """#metadata[metadata['lab_name']=='hematocrit']"""
    
        mrkd8 = """The cell below should be used to update outlier bounds. The third argument defines the minimum (lower) bound and the fourth argument defines the maximum (upper) bound. The values for the upper and lower bounds should be specified in the standard units."""
        code9 = """#updated_metadata = update_outlier_bounds(metadata, 'hematocrit', None, 1000)"""        
        
        mrkd9 = """This cell outputs the updated minimum and maximum value for the lab measurement defined in the metadata table."""
        code10 = """#updated_metadata[updated_metadata['lab_name']=='hematocrit']""" 
        
        nb_code = [nbf.v4.new_markdown_cell(mrkd_intro)
                    , nbf.v4.new_code_cell(code1), nbf.v4.new_markdown_cell(mrkd1)
                    , nbf.v4.new_code_cell(code2), nbf.v4.new_markdown_cell(mrkd2)
                    , nbf.v4.new_code_cell(code3), nbf.v4.new_markdown_cell(mrkd3)
                    , nbf.v4.new_code_cell(code4), nbf.v4.new_markdown_cell(mrkd4)
                    , nbf.v4.new_code_cell(code5), nbf.v4.new_markdown_cell(mrkd5)
                    , nbf.v4.new_code_cell(code6), nbf.v4.new_markdown_cell(mrkd6)
                    , nbf.v4.new_code_cell(code7), nbf.v4.new_markdown_cell(mrkd7)
                    , nbf.v4.new_code_cell(code8), nbf.v4.new_markdown_cell(mrkd8)
                    , nbf.v4.new_code_cell(code9), nbf.v4.new_markdown_cell(mrkd9)
                    , nbf.v4.new_code_cell(code10)]    
    
############################    
    else:
        dtype_mk = datatype.title().replace('Pfhh', 'PFHH').replace('physical_measurement', '')
        cid_mk = str(concept_id).replace('-', ' ')

        mrkd_intro = f"""
<div class='alert alert-info' style='font-family:Arial; font-size:20px' ><center><font size="6"> {cid_mk} {dtype_mk} Phenotype Summary</font></center></div>

This notebook includes graphical summaries of cases and controls (see '0_ReadMe') for the {cid_mk} {dtype_mk} phenotype. 

**Note:**
- If you have not already, **run the set up in the '0_ReadMe' once** before running the notebooks in this workspace.
- This notebook has been automatically generated. If you wish to rerun it, you may need to choose the R kernel (top menu).
"""   
        mrkd_setup = f"""# Import summary functions
source('cat_data_summary_functions.R'); source('utilities.R')"""

#######
        mrkd_cat_sum_intro = f"""# Summary    
Below are summaries of cases and controls for the {cid_mk} {datatype} phenotype:
- Overall summaries by 1) age, 2) sex at birth, and 3) ancestry.
- Detailed summaries by 1) sex at birth and ancestry and 2) age and ancestry. """ 
    
        code_cat = f"""categorical_data_summary(concept_id = '{concept_id}', filename = '{phe_filename}'
                                                , datatype = '{datatype}', map_concept_name = FALSE)"""

#######
        mrkd_cont_sum_intro = f"""# Summary   
Below are summaries for {cid_mk} {dtype_mk}.
- A table of descriptive statistics.
- An histogram and a boxplot.
- Demographic counts: 1) age, sex at birth, ancestry, 4) sex at birth and ancestry and 5) age and ancestry."""
    
        code_cont = f"""pm_data_summary(concept_name = '{concept_id}')"""
    
################################################# create notebook  ##################################################
        r_code_dd = {"drug":[mrkd_setup, mrkd_cat_sum_intro, code_cat]
                     ,"phecode":[mrkd_setup, mrkd_cat_sum_intro, code_cat]
                     ,"pfhh":[mrkd_setup, mrkd_cat_sum_intro.replace('FALSE', 'TRUE'),code_cat]
                     ,"physical_measurement":[mrkd_setup.replace('cat_data_summary_functions.R', 'cont_data_summary_functions.R')
                                          , mrkd_cont_sum_intro, code_cont]}     
        ##########
        nb_code = [nbf.v4.new_markdown_cell(mrkd_intro), nbf.v4.new_code_cell(r_code_dd[datatype][0])
                       , nbf.v4.new_markdown_cell(r_code_dd[datatype][1]), nbf.v4.new_code_cell(r_code_dd[datatype][2])]
        
    return nb_code
        
def get_notebook_name(datatype, concept_id):
    if 'measurement' in datatype.replace('_',' '):
        pm_name = str(concept_id).lower().replace('-','_')
        notebook_name = f"{pm_name}_summary.ipynb"
    elif 'lab' in datatype.replace('_',' '): notebook_name = f"measurement_{str(concept_id)}.ipynb"
    else: notebook_name = f"{datatype}_{str(concept_id)}_summary.ipynb"
    
    return notebook_name

def generate_notebook(concept_id, datatype, notebook_name, phe_filename):
    
    # initiate notebook
    nb = nbf.v4.new_notebook()    
    # write notebook code
    nb['cells'] = get_notebook_code(datatype, concept_id, phe_filename)
    # Save notebook to bucket
    bucket = os.getenv('WORKSPACE_BUCKET')
    nbf.write(nb, notebook_name)
    os.system(f"gsutil cp {notebook_name} {bucket}/notebooks/")
    

def get_kernel(kernel):
    return 'ir' if kernel.lower() == 'r' else 'python3'

def run_notebook(NOTEBOOK_TO_RUN, KERNEL = 'r'):

    KERNEL_NAME = get_kernel(KERNEL)    
    OUTPUT_NOTEBOOK = NOTEBOOK_TO_RUN
    bucket = os.getenv('WORKSPACE_BUCKET')
    
    with open(NOTEBOOK_TO_RUN) as f_in:
        nb = nbf.read(f_in, as_version=4)
        ep = ExecutePreprocessor(timeout=-1, kernel_name=KERNEL_NAME)
        out = ep.preprocess(nb, {'metadata': {'path': ''}})
    with open(OUTPUT_NOTEBOOK, mode='w', encoding='utf-8') as f_out:
        nbf.write(nb, f_out)
        os.system(f"gsutil cp {OUTPUT_NOTEBOOK} {bucket}/notebooks/{OUTPUT_NOTEBOOK}")
        
    print(f'Ran and saved {OUTPUT_NOTEBOOK}.')
    
def generate_and_run_all_nb(datatype, phe_filename):
    #combines all functions to genrate and run all the notebooks for the workspaces (for every phenotype)
    concept_ids =  pd.read_csv(phe_filename, nrows=0).drop('person_id', axis =1).columns
    start = datetime.now()
    n = 1
    print(n)
    for concept_id in concept_ids:
        notebook_name = get_notebook_name(datatype, concept_id)
        generate_notebook(concept_id, datatype, notebook_name, phe_filename)
        run_notebook(notebook_name)
        n = n+1

    print('DONE.')
    end = datetime.now()
    totaltime = end-start
    print(totaltime)


###
BUCKET = %env WORKSPACE_BUCKET
B_NB = BUCKET+'/notebooks'

drug_bucket = 'gs://fc-secure-15736bb7-7fa3-4bf8-8870-8f043b195ca5'
pfhh_bucket = 'gs://fc-secure-b00c0e03-c1e7-47f3-b069-6760e9ac8081'
phex_bucket = 'gs://fc-secure-8f84b9e6-eb17-4d0f-be14-c710ae19795d'
pm_bucket = 'gs://fc-secure-264cb142-0acd-4173-8fda-a256d09f3149'
lab_bucket = 'gs://fc-secure-93cc8e81-154f-4d54-ab68-4af358dfa5da'

!gsutil -m cp utilities.py {drug_bucket}/notebooks/code
!gsutil -m cp utilities.py {pfhh_bucket}/notebooks/code
!gsutil -m cp utilities.py {phex_bucket}/notebooks/code
!gsutil -m cp utilities.py {pm_bucket}/notebooks/code
!gsutil -m cp utilities.py {lab_bucket}/notebooks/code

# %%writefile cat_data_summary_functions.R

# # for Survey, drug or phecode Data
# ##################################################### SET UP ##############################################################
# package_list <- c('bigrquery','tidyverse','dplyr','janitor', 'data.table')
# for (pkg in package_list) if(!pkg %in% installed.packages()) {install.packages(pkg, quiet = T)}
    
# library(bigrquery, warn.conflicts = F, quietly = T)
# library(tidyverse, warn.conflicts = F, quietly = T)
# library(janitor, warn.conflicts = F, quietly = T)
# library(dplyr, warn.conflicts = F, quietly = T)
# library(data.table)
# library(readr)
# options(dplyr.summarise.inform = FALSE)

# barplot <- function(df, plot_title, n_col, X, Y = `Percentage`
#                              , Fill = "Cases", Facet = "Cases"
#                              , fill_palette = "Pastel1"
#                              , facet_nrow = 1, facet_col = 2, base_text_size =9, w = 14, h = 5){
    
#     df["Percentage"] <- round((df[n_col]/df['numerator'])*100,2)
#     df["Label"] <- paste0(format(df[[n_col]], big.mark=","), ' (', df$Percentage, '%)')    
#     if (!is.null(Facet)){df["Facet"] <- paste0('\n\n',df[[Facet]], '\nN=', format(df$numerator, big.mark=","), ' (100%)')}
    
#     ###### Plot #####
#     options(repr.plot.width = w, repr.plot.height = h) #~1.5 per bar
#     p <- ggplot(data=df, aes(x={{X}}, y={{Y}}, fill = {{Fill}}
#                             )) +
#             geom_bar(stat="identity", position = 'dodge2') +
#             geom_text(aes(label= Label), hjust="inward", vjust = 0.5, size=(base_text_size/3)+1
#                       , position = position_dodge(width = 0.9)) +
#             labs(x = '', y = '', title = plot_title) +
#             theme_minimal()+ 
#             scale_fill_brewer(palette = fill_palette) +
#             theme(axis.text.x = element_blank(), axis.text.y = element_text(size = base_text_size+4)
#                   , legend.title = element_blank()
#                   , legend.text = element_text(size = base_text_size+5)
#                   , legend.position = "top", legend.box = "horizontal"
#                   , plot.title = element_text(hjust = 0.5)
#                   , title = element_text(size = base_text_size+4)) +
#             coord_flip()
    
#     if (!is.null(Facet)){
#         p <- p+facet_wrap(~Facet, nrow = facet_nrow, ncol = facet_col)+
#                 theme(strip.text.x = element_text(size = base_text_size+6))+theme(legend.position = "none")
#             }
#     return(p)
#     }

# wrangle_cat_data <- function(data_df){
#     colnames(data_df) = c('person_id', 'Cases')
#     data_df$Cases[data_df$Cases ==TRUE]<-'Cases'
#     data_df$Cases[data_df$Cases ==FALSE]<-'Controls'  
#     data_df <- data_df[!is.na(data_df$Cases),] # NEW removing NAs
#     return(data_df)
#     }

# count_by <- function(data_df, var, var_df){    

#     var_df = unique(var_df[c('person_id',var)])    
#     merged_df = inner_join(data_df, var_df, by = 'person_id') 
#     counts_df <- merged_df %>% dplyr::group_by(merged_df[var], Cases) %>% 
#                     dplyr::summarise('n_pids' = n_distinct(person_id))
    
#     counts_df <- cbind(counts_df, numerator=NA)
#     counts_df$numerator[counts_df$Cases =='Cases'] <- n_distinct(merged_df$person_id[merged_df$Cases =='Cases'])
#     counts_df$numerator[counts_df$Cases =='Controls'] <- n_distinct(merged_df$person_id[merged_df$Cases =='Controls'])
#     counts_df$numerator[counts_df$Cases =='NA'] <- n_distinct(merged_df$person_id[merged_df$Cases =='NA'])
    
#     return(counts_df)
#     }

# categorical_data_summary <- function(concept_id, filename, datatype, map_concept_name = TRUE){
    
#     # LOAD Data   
#     ## Demographics Data
#     demographics_df <- fread("demographics_table.csv", data.table = FALSE) #returns dataframe -faster than read_csv
#     ## Ancestry Data
#     ancestry_df = fread('ancestry.csv', data.table = FALSE) 
#     ancestry_df$ancestry <- paste(ancestry_df$ancestry, " ")
#     data_df = fread(filename, select  = c('person_id', concept_id), data.table = FALSE)

#     # TRANSFORM DATA
#     Data_df <- wrangle_cat_data(data_df)
#     age_count_df <- count_by(Data_df, var= 'age_group', var_df = demographics_df)
    
#     sex_count_df <- count_by(Data_df, var= 'sex_at_birth', var_df = demographics_df)
#     ancestry_count_df <- count_by(Data_df, var= 'ancestry', var_df = ancestry_df)
#     sex_and_ancestry_counts_df <- count_by(Data_df, var= c('ancestry','sex_at_birth')
#                                            , var_df = inner_join(ancestry_df[c('person_id','ancestry')]
#                                                                  , demographics_df[c('person_id','sex_at_birth')]
#                                                                  , by = 'person_id'))

#     age_and_ancestry_counts_df <- count_by(Data_df, var= c('ancestry','age_group')
#                                            , var_df = inner_join(ancestry_df[c('person_id','ancestry')]
#                                                                  , demographics_df[c('person_id','age_group')]
#                                                                  , by = 'person_id'))

    
#     ############################################################################################
#     if (map_concept_name == TRUE){
#         Map <- download_data(str_glue("SELECT concept_name FROM `concept` \
#                                         WHERE concept_id = {concept_id} or concept_code = '{concept_id}'"))
#         if (nrow(Map) == 0){concept_name = ''} 
#         else {concept_name <- Map$concept_name; concept_name <- str_glue(' - {concept_name}')}
#       } else {concept_name = ''}   
#     concept_id = str_glue("{concept_id}{concept_name}")
#     datatype = toupper(datatype)
    
#     ##################################################PLOT SUMMARIES#######################################################
#     #'OVERALL SUMMARIES BY AGE, SEX AT BIRTH AND ANCESTRY
    
#     n = 1
#     View(barplot(age_count_df, n_col = "n_pids", X = `age_group`, h = 6
#                 , plot_title= str_glue('OVERALL SUMMARIES BY AGE, SEX AT BIRTH AND ANCESTRY\n\n\nFigure {n}: Age at CDR')))
#     n = n+1
#     View(barplot(sex_count_df, n_col = "n_pids", X = `sex_at_birth`, plot_title= str_glue('\nFigure {n}: Sex at Birth')
#                     , fill_palette = "Blues"))#+theme(legend.position = "none")
#     n = n+1
#     View(barplot(ancestry_count_df, n_col = "n_pids", X = `ancestry`, plot_title= str_glue('\nFigure {n}: Ancestry')
#                      , fill_palette = "Greens"))
    
    
#     # DETAILED SUMMARIES BY SEX AT BIRTH AND ANCESTRY
#     sex_at_births = unique(sex_and_ancestry_counts_df$sex_at_birth)
#     n = n+1
#     var_sex1 = sex_at_births[1]
#     df_sex1 = sex_and_ancestry_counts_df[sex_and_ancestry_counts_df$sex_at_birth == var_sex1,]
#     View(barplot(df_sex1, n_col = "n_pids", X = `ancestry`, Fill = `sex_at_birth`, fill_palette = "Purples"
#                      , plot_title= str_glue('DETAILED SUMMARIES BY SEX AT BIRTH AND ANCESTRY\n\n\nFigure {n}: {var_sex1} (Sex at Birth) by Ancestry')
#                     , facet_nrow = 1, facet_col = 2))   
#     for (var in sex_at_births[-1]){
#         n = n+1
#         df_var = sex_and_ancestry_counts_df[sex_and_ancestry_counts_df$sex_at_birth == var,]
#         View(barplot(df_var, n_col = "n_pids", X = `ancestry`, Fill = `sex_at_birth`, fill_palette = "Purples"
#                      , plot_title= str_glue('\nFigure {n}: {var} by Ancestry')
#                     , facet_nrow = 1, facet_col = 2))
#         }

    
#     # DETAILED SUMMARIES BY AGE AND ANCESTRY
#     ages = unique(age_and_ancestry_counts_df$age_group)
#     n = n+1
#     var_age1 = ages[1]
#     df_age1 = age_and_ancestry_counts_df[age_and_ancestry_counts_df$age_group == var_age1,]
#     View(barplot(df_age1, n_col = "n_pids", X = `ancestry`, Fill = `age_group`, fill_palette = "Pastel2"
#                     , plot_title= str_glue('DETAILED SUMMARIES BY AGE AT CDR AND ANCESTRY\n\n\nFigure {n}: {var_age1} Years Old by Ancestry')
#                         , facet_nrow = 1, facet_col = 2)) 
#     for (var in ages[-1]){
#         n = n+1
#         df_var = age_and_ancestry_counts_df[age_and_ancestry_counts_df$age_group == var,]
#         View(barplot(df_var, n_col = "n_pids", X = `ancestry`, Fill = `age_group`, fill_palette = "Pastel2"
#                          , plot_title= str_glue('\nFigure {n}: {var} Years Old by Ancestry')
#                               , facet_nrow = 1, facet_col = 2))
#         }
#     }


!gsutil -m cp cat_data_summary_functions.R {B_NB}/code/
!gsutil -m cp cat_data_summary_functions.R {pfhh_bucket}/notebooks/code
!gsutil -m cpcat_data_summary_functions.R {phex_bucket}/notebooks/code