
###################################################### SET UP ##############################################################
package_list <- c('bigrquery','tidyverse','dplyr','janitor', 'data.table')
for (pkg in package_list) if(!pkg %in% installed.packages()) {install.packages(pkg, quiet = T)}

library(bigrquery, warn.conflicts = F, quietly = T)
library(tidyverse, warn.conflicts = F, quietly = T)
library(janitor, warn.conflicts = F, quietly = T)
library(dplyr, warn.conflicts = F, quietly = T)
library(data.table)
options(dplyr.summarise.inform = FALSE)

# the current workspace's dataset variable (to use in the query)
# dataset <- Sys.getenv('WORKSPACE_CDR')
# billing_project <- Sys.getenv('GOOGLE_PROJECT')
# my_bucket <- Sys.getenv('WORKSPACE_BUCKET')

# helper function to make the download easier
download_data <- function(query) {
    tb <- bq_project_query(Sys.getenv('GOOGLE_PROJECT'), page_size = 25000,
                           query = query, default_dataset = Sys.getenv('WORKSPACE_CDR'))
    bq_table_download(tb)
}

read_csv_cols_from_bucket <- function(directory = 'notebooks/phenotype_data/', name_of_file_in_bucket, concept_id
                                      , remove_dot = 'no'){   
    #reads person_id and concept id columns in a df from the input csv file in teh bucket
    my_bucket = Sys.getenv('WORKSPACE_BUCKET')
    
    if (remove_dot == 'no'){
        system(paste0("gsutil cp ", my_bucket, "/", directory, name_of_file_in_bucket, " ."), intern=T)              
        } else {system(paste0("gsutil cp ", my_bucket, "/", directory, name_of_file_in_bucket), intern=T)}
        
    my_dataframe  <- fread(name_of_file_in_bucket, select = c("person_id", str_glue("{concept_id}"))
                           , data.table = FALSE) #returns dataframe
    return(my_dataframe)
    }

read_csv_from_bucket <- function(directory = 'notebooks/phenotype_data/', name_of_file_in_bucket, remove_dot = 'no'){
    my_bucket = Sys.getenv('WORKSPACE_BUCKET')
    
    if (remove_dot == 'no'){
        system(paste0("gsutil cp ", my_bucket, "/", directory, name_of_file_in_bucket, " ."), intern=T)              
        } else {system(paste0("gsutil cp ", my_bucket, "/", directory, name_of_file_in_bucket), intern=T)}
        
    my_dataframe  <- fread(name_of_file_in_bucket, data.table = FALSE) #returns dataframe
    return(my_dataframe)
    }
    
