# R code to extract, analyze and create summary reports, including visualizations, on the labratory measurement data.

package_list <- c('bigrquery','tidyverse','dplyr','janitor','data.table')
for (pkg in package_list) if(!pkg %in% installed.packages()) {install.packages(pkg, quiet = T)}

library(bigrquery, warn.conflicts = F, quietly = T)
library(tidyverse, warn.conflicts = F, quietly = T)
library(janitor, warn.conflicts = F, quietly = T)
library(dplyr, warn.conflicts = F, quietly = T)
library(data.table, warn.conflicts = F, quietly = T)
options(dplyr.summarise.inform = FALSE)

bq <- function(query) {
    bq_table_download(bq_project_query(Sys.getenv('GOOGLE_PROJECT'), page_size = 25000,
                                       query=query, default_dataset = Sys.getenv('WORKSPACE_CDR')))
    }


lab_data <- function(ancestor_cid){
    
    df <- bq(str_glue("
                SELECT DISTINCT person_id
                , measurement_concept_id
                , LOWER(cm.concept_name) as measurement_concept_name, LOWER(cam.concept_name) as ancestor_concept_name
                , value_as_number
                , range_low, range_high
                , unit_concept_id, LOWER(u.concept_name) as unit_concept_name
                , value_as_concept_id, LOWER(cv.concept_name) as value_as_concept_name
                , operator_concept_id, LOWER(co.concept_name) as operator_concept_name 
                , measurement_datetime --, measurement_date
                , src_id ###, MAX(measurement_date) AS most_recent_measurement_date
                
                #see note 2 - only use those when the standard concepts are not available/do not make sense
                , measurement_source_concept_id, LOWER(measurement_source_value) AS measurement_source_value
                , LOWER(unit_source_value) AS unit_source_value
                #, CASE WHEN value_as_number IS NOT NULL THEN 'numeric measurement' 
                       #WHEN value_as_number IS NULL and (value_as_concept_id IS NOT NULL OR value_as_concept_id !=0)
                            #THEN 'categorical measurement' 
                      # ELSE 'other' END as measurement_category

            FROM `measurement` m 
            JOIN `measurement_ext` m_ext ON m.measurement_id = m_ext.measurement_id
            JOIN `concept_ancestor` ON descendant_concept_id = measurement_concept_id
            JOIN `concept` as cm on cm.concept_id = measurement_concept_id
            JOIN `concept` as cam on cam.concept_id = ancestor_concept_id
            LEFT JOIN (SELECT c2.concept_name, c1.concept_id 
                    FROM `concept` c1 JOIN `concept` c2 on c1.concept_name = c2.concept_code
                    WHERE c1.domain_id = 'Meas Value') as cv on cv.concept_id = value_as_concept_id            
            LEFT JOIN `concept` as co on co.concept_id = operator_concept_id
            LEFT JOIN `concept` as u on u.concept_id = unit_concept_id
                      
            WHERE LOWER(m_ext.src_id) LIKE '%ehr%' AND ancestor_concept_id IN ({ancestor_cid})
                      AND value_as_number IS NOT NULL
            
            "))
    filename = str_glue('measurement_{ancestor_cid}.csv')
    print(str_glue("\nFinal output will be saved to bucket as {filename}\n"))
    
    
    return(df)
    }
                      
simple_boxplot<- function(measurement_df, title = '', ancestor_concept_name, w = 20, h = 8){   
    
     #units = paste0(unique(measurement_df$unit_concept_name),collapse = ', ')
    #title = paste0(str_to_upper(title), "\n", str_glue("Units: {units}"))
    
    options(repr.plot.width = w, repr.plot.height = h)
    
    measurement_df$scr_id <- gsub('EHR site ', '#', measurement_df$src_id) 
    boxplot(value_as_number~src_id, data=measurement_df,main=title, xlab="EHR Site"
            , ylab= str_glue("{ancestor_concept_name} Values"))
    
    }
                      

simple_histogram<- function(measurement_df, value_col = 'value_as_number'
                            ,title = '', ancestor_concept_name, w = 15, h = 8){   
    
    #units = paste0(unique(measurement_df$unit_concept_name),collapse = ', ')
    #title = paste0(str_to_upper(title), "\n", str_glue("Units: {units}"))
    
    options(repr.plot.width = w, repr.plot.height = h)
    hist(measurement_df[[value_col]], main = title, xlab= str_glue("{ancestor_concept_name} Values"))
    
    }

stats_table <- function(df_measurement){
    
    #value_column = 'value_as_number'
    
    #df_measurement <- drop_na(df_measurement)
    stats_df <- df_measurement %>%
            dplyr::group_by(src_id) %>%
            dplyr::summarize('Mean' = mean(value_as_number, na.rm = TRUE)
                             ,'Min' = min(value_as_number, na.rm = TRUE)
                             ,'Max' = max(value_as_number, na.rm = TRUE)
                             ,'1% Quantile' = quantile(value_as_number, 0.01, na.rm = TRUE)
                             ,'2% Quantile' = quantile(value_as_number, 0.02, na.rm = TRUE)
                             ,'25th Quantile' = quantile(value_as_number, 0.25, na.rm = TRUE)
                             ,'50th Quantile (Median)' = quantile(value_as_number, 0.5, na.rm = TRUE)
                             ,'75th Quantile' = quantile(value_as_number, 0.75, na.rm = TRUE)
                             ,'98th Quantile' = quantile(value_as_number, 0.98, na.rm = TRUE)
                             ,'99th Quantile' = quantile(value_as_number, 0.99, na.rm = TRUE)
                             ,'Standard Deviation' = sd(value_as_number, na.rm = TRUE)
                            ) %>%
            rename('EHR Site' = src_id)
    
    return(stats_df)
}
                      
final_dataframe <- function(measurement_df, concept_id){
    latest_df <- measurement_df[c('person_id', 'measurement_datetime', 'value_as_number', 'unit_concept_name')] %>%
                        dplyr::group_by(person_id) %>%
                        filter(measurement_datetime == max(measurement_datetime)) %>%
                        summarize(value_as_number = paste0(value_as_number,  collapse = ', ')
                                 , unit_concept_name = paste0(unit_concept_name,  collapse = ', ')) %>%
                        rename(latest_value = value_as_number, latest_value_unit = unit_concept_name)

    count_df <- drop_na(measurement_df[c('person_id','value_as_number')]) %>%
                dplyr::group_by(person_id) %>%
                dplyr::summarize('values_count' = n_distinct(value_as_number, na.rm = TRUE))

    output_df <- measurement_df[c('person_id','value_as_number')] %>%
                dplyr::group_by(person_id) %>%
                dplyr::summarize('Mean' = mean(value_as_number, na.rm = TRUE)
                                 ,'Min' = min(value_as_number, na.rm = TRUE)
                                 ,'Max' = max(value_as_number, na.rm = TRUE)
                                 ,'1% Quantile' = quantile(value_as_number, 0.01, na.rm = TRUE)
                                 ,'2% Quantile' = quantile(value_as_number, 0.02, na.rm = TRUE)
                                 ,'25th Quantile' = quantile(value_as_number, 0.25, na.rm = TRUE)
                                 ,'50th Quantile (Median)' = quantile(value_as_number, 0.5, na.rm = TRUE)
                                 ,'75th Quantile' = quantile(value_as_number, 0.75, na.rm = TRUE)
                                 ,'98th Quantile' = quantile(value_as_number, 0.98, na.rm = TRUE)
                                 ,'99th Quantile' = quantile(value_as_number, 0.99, na.rm = TRUE)
                                 ,'Standard Deviation' = sd(value_as_number, na.rm = TRUE)
                                ) %>%
                merge(latest_df) %>%
                merge(count_df)
    
    #### final dataframe output
    filename = str_glue('measurement_{concept_id}.csv')
    write_csv(output_df, filename)
    
    my_bucket = Sys.getenv('WORKSPACE_BUCKET')
    directory = 'notebooks/all_x_all/'
    
    system(paste0("gsutil cp ", filename, " ", my_bucket, "/", directory, filename), intern=T)
    system(paste0("gsutil ls ", my_bucket, "/", directory, filename), intern=T)
    

    return(output_df) 
    }
                      
                      
lab_data_summary <- function(concept_id){
    df_measurement <- lab_data(ancestor_cid = concept_id)
    
    ancestor_concept_name = unique(df_measurement$ancestor_concept_name)
    title = str_to_upper(str_glue('\n\n\n~~~~~~~~~EHR {ancestor_concept_name} Summary & Distributions\n\n~~~~~~~~~~'))
    
    cat(title)
    n_pids = n_distinct(df_measurement$person_id)
    print(str_glue('N Pids :{n_pids}'))                     

    #cat("\n\n")
    df_measurement$src_id<- gsub('EHR site ', '#', df_measurement$src_id)
    
    simple_histogram(df_measurement, ancestor_concept_name = ancestor_concept_name)
    
    simple_boxplot(df_measurement, ancestor_concept_name = ancestor_concept_name)
    
    #lab_summary(df_measurement, ancestor_concept_name)   
    stats_df <- stats_table(df_measurement)
    final_df <- final_dataframe(measurement_df = df_measurement, concept_id = concept_id)
    stats_df
 }
                      