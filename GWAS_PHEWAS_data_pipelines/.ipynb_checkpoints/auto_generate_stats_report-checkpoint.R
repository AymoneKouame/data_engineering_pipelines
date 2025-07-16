# Code to Auto Generate a summary report notebook for each Phenotype

package_list <- c('tidyverse','markdown', 'rmd2jupyter', 'data.table')
for (pkg in package_list) if(!pkg %in% installed.packages()) {install.packages(pkg)}
if(!'rmd2jupyter' %in% installed.packages()) {devtools::install_github("mkearney/rmd2jupyter")}
    
library(tidyverse)
library(markdown)
library(rmd2jupyter)
library(data.table)


########## Function/Code to Generate a notebook for each concept ID/column name
cat_data_code <- "
    ################################################## CODE ##############################################

    # Set Up

    ## Loading packages and custom functions for AllxAll Phenotypes Summaries

    source_code_filename <- 'allxall_cat_data_summary_functions.R'
    system(paste0('gsutil cp ', Sys.getenv('WORKSPACE_BUCKET'), '/notebooks/all_x_all/', source_code_filename,  ' ./'), intern=T)
    source(source_code_filename)

    # ################################################## SUMMARY ##############################################
    categorical_data_summary(concept_id = '{concept_id}', name_of_file_in_bucket = '{name_of_file_in_bucket}'
                             , datatype = '{datatype}', map_concept_name = FALSE)
"

cont_data_code <- "
    ################################################## CODE ##############################################

    # Set Up

    ## Loading packages and custom functions for AllxAll Phenotypes Summaries

    source_code_filename <- 'allxall_cat_data_summary_functions.R'
    system(paste0('gsutil cp ', Sys.getenv('WORKSPACE_BUCKET'), '/notebooks/all_x_all/', source_code_filename,  ' ./'), intern=T)
    source(source_code_filename)

    # ################################################## SUMMARY ##############################################
    continuous_data_summary(concept_id = '{concept_id}', name_of_file_in_bucket = '{name_of_file_in_bucket}'
                             , datatype = '{datatype}', map_concept_name = FALSE)
"

pm_data_code <- "  
    ################################################## CODE ##############################################

    # Set Up

    ## Loading packages and custom functions for AllxAll Phenotypes Summaries

    source_code_filename <- 'allxall_pm_summary_functions.R'
    system(paste0('gsutil cp ', Sys.getenv('WORKSPACE_BUCKET'), '/notebooks/all_x_all/', source_code_filename,  ' ./'), intern=T)
    source(source_code_filename)

    # ################################################## SUMMARY ##############################################
    pm_data_summary(concept_id = '{concept_id}')
"

lab_data_code <- "
    ################################################## CODE ##############################################

    # Set Up

    ## Loading packages and custom functions for AllxAll Phenotypes Summaries

    source_code_filename <- 'allxall_lab_summary_functions.R'
    system(paste0('gsutil cp ', Sys.getenv('WORKSPACE_BUCKET'), '/notebooks/all_x_all/', source_code_filename,  ' ./')
           , intern=T)
    source(source_code_filename)

    # ################################################## SUMMARY ##############################################
    lab_data_summary(concept_id = '{concept_id}')

    "

filename_dd <- c( #datatype = c('name_of_file_in_bucket', code)
      "drug"= "r_drug_table.csv",
      "phecode" = "mcc2_phecode_table.csv",
      "pfhh"= "pfhh_survey_table.csv",
      "physical_measurement"= "physical_measurement_table.csv",
      #"lab_measurement"= "physical_measurement_table.csv"
    )

code_dd <- c( #datatype = c('name_of_file_in_bucket', code)
      "drug"= cat_data_code,
      "phecode" = cat_data_code,
      "pfhh"= gsub('FALSE', 'TRUE', cat_data_code),
      "physical_measurement"= pm_data_code,
      #"lab_measurement" = lab_data_code
    )

read_csv_from_bucket <- function(directory = 'notebooks/all_x_all/', name_of_file_in_bucket, remove_dot = 'no'){
    
    #reads person_id and concept id columns in a df from the input csv file in teh bucket
    #if input csv file is already read from the bucket to the env, it simply reads the person_id and concept id columns in a df
    my_bucket = Sys.getenv('WORKSPACE_BUCKET')
    
    if (file.exists(name_of_file_in_bucket)){my_dataframe  <- fread(name_of_file_in_bucket)
    } else {
            if (remove_dot == 'no'){
                    system(paste0("gsutil cp ", my_bucket, "/", directory, name_of_file_in_bucket, " ."), intern=T)              
            } else {system(paste0("gsutil cp ", my_bucket, "/", directory, name_of_file_in_bucket), intern=T)}
        
            my_dataframe  <- fread(name_of_file_in_bucket)
            }
    my_dataframe  <- as.data.frame(my_dataframe)
    return(my_dataframe)
    }

write_r_code_to_notebook <- function(datatype, concept_id, r_code, name_of_file_in_bucket){
    
    cid_r_code <- str_glue(r_code)
    if (datatype == 'physical_measurement'){
        pm_name <-tolower(gsub('-','_', concept_id)); notebook_name = str_glue("{pm_name}_summary")
    } else {notebook_name = str_glue("{datatype}_{concept_id}_summary")}

    # Save string to .R file
    fileConn<-file(str_glue("{notebook_name}.R"))
    writeLines(cid_r_code, fileConn)
    close(fileConn)
    
    # Transform .R to .Rmd
    knitr::spin(str_glue("{notebook_name}.R"), format ="Rmd", knit = FALSE)# to transform it to Rmd
    
    # Transform .RMD to .ipynb
    rmd2jupyter(str_glue("{notebook_name}.Rmd"))
    
    # Remove the .Rmd and .R files/ keep the .ipynb file
    file.remove(str_glue("{notebook_name}.R"))
    file.remove(str_glue("{notebook_name}.Rmd"))
    #print(str_glue("{notebook_name}.ipynb created"))
    
    }

generate_r_notebooks <- function(datatype){
    
    name_of_file_in_bucket = as.character(filename_dd[datatype])
    r_code = as.character(code_dd[datatype])

    df_input <- read_csv_from_bucket(name_of_file_in_bucket = name_of_file_in_bucket)
    concept_ids <- colnames(select(df_input,-c('person_id')))

    start = Sys.time()
    print(start)

    n = 1
    for (concept_id in concept_ids) {
        print(n)
        write_r_code_to_notebook(datatype, concept_id, r_code, name_of_file_in_bucket)
        n = n+1
        }
    print('DONE.')
    end = Sys.time()
    totaltime = end-start
    print(totaltime)    
    }

## How to un
generate_r_notebooks(datatype = 'drug')
generate_r_notebooks(datatype = 'phecode')
generate_r_notebooks(datatype = 'pfhh')
generate_r_notebooks(datatype = 'lab_measurement')
generate_r_notebooks(datatype = 'physical_measurement')
generate_r_notebooks(datatype = 'lab_data')
#generate_r_notebooks(datatype = 'physical_measurement')