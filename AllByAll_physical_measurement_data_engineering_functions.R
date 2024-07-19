
##################################################### SET UP ##############################################################
package_list <- c('bigrquery','tidyverse','dplyr','janitor','data.table')
for (pkg in package_list) if(!pkg %in% installed.packages()) {install.packages(pkg, quiet = T)}

library(bigrquery, warn.conflicts = F, quietly = T)
library(tidyverse, warn.conflicts = F, quietly = T)
library(janitor, warn.conflicts = F, quietly = T)
library(dplyr, warn.conflicts = F, quietly = T)
library(data.table, warn.conflicts = F, quietly = T)
options(dplyr.summarise.inform = FALSE)

read_csv_cols_from_bucket <- function(name_of_file_in_bucket, concept_id, my_bucket = 'gs://allxall-phenotypes'
                                      , directory = 'data/', remove_dot = 'no'){
    
    #reads person_id and concept id columns in a df from the input csv file in teh bucket
    #if input csv file is already read from the bucket to the env, it simply reads the person_id and concept id columns in a df
    
    if (file.exists(name_of_file_in_bucket)){my_dataframe  <- fread(name_of_file_in_bucket
                                                                  , select = c("person_id", str_glue("{concept_id}")))
    } else {
            if (remove_dot == 'no'){
                    system(paste0("gsutil cp ", my_bucket, "/", directory, name_of_file_in_bucket, " ."), intern=T)              
            } else {system(paste0("gsutil cp ", my_bucket, "/", directory, name_of_file_in_bucket), intern=T)}
        
            my_dataframe  <- fread(name_of_file_in_bucket, select = c("person_id", str_glue("{concept_id}")))
            }

    return(my_dataframe)
    }

read_csv_from_bucket <- function(name_of_file_in_bucket, my_bucket = 'gs://allxall-phenotypes'
                                 , directory = 'data/', remove_dot = 'no'){
    
    #reads person_id and concept id columns in a df from the input csv file in teh bucket
    #if input csv file is already read from the bucket to the env, it simply reads the person_id and concept id columns in a df
    
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

                      
barplot <- function(df, plot_title, n_col, X, Y = `Percentage`, Fill = "grey", Facet = "Cases"
                    , facet_nrow = NULL, facet_col = NULL, base_text_size =12, w = 14, h = 8){
    
    df["Percentage"] <- round((df[n_col]/df['numerator'])*100,2)
    df["Label"] <- paste0(format(df[[n_col]], big.mark=","), ' (', df$Percentage, '%)')    

    if (!is.null(Facet)){df["Facet"] <- paste0(df[[Facet]], 'N=', format(df$numerator, big.mark=","), ' (100%)')}
    
    ###### Plot #####

    options(repr.plot.width = w, repr.plot.height = h) #~1.5 per bar
    p <- ggplot(data=df, aes(x={{X}}, y={{Y}}#, fill = {{Fill}}
                            )) +
            geom_bar(stat="identity", position = 'dodge2', fill = Fill) +
            geom_text(aes(label= Label), hjust="inward", vjust = 0.5, size=(base_text_size/3)+1
                      , position = position_dodge(width = 0.9)) +
            labs(x = '', y = '', title = plot_title) +
            theme_minimal()+ 
            theme(axis.text.x = element_blank(), axis.text.y = element_text(size = base_text_size+4)
                  , legend.title = element_blank()
                  , legend.text = element_text(size = base_text_size+5)
                  , legend.position = "top", legend.box = "horizontal"
                  , plot.title = element_text(hjust = 0.5)
                  , title = element_text(size = base_text_size+4)) +
            coord_flip()
  
    if (!is.null(Facet)){
        p <- p+facet_wrap(~Facet, nrow = facet_nrow, ncol = facet_col)+
                theme(strip.text.x = element_text(size = base_text_size+6))+theme(legend.position = "none")
            }
    return(p)
    }

##################################################DATA ############################################################
#system2('gsutil',args = c('cp','gs://fc-aou-preprod-datasets-controlled/v7/wgs/without_ext_aian_prod/vds/aux/ancestry/delta_v1_gt_no_ext_aian_gq0_prod.ancestry_preds.tsv','./ancestry.tsv'))
ancestry_data_path <- 'gs://prod-drc-broad/aou-wgs-delta-aux_gq0/ancestry/delta_v1_gt_no_ext_aian_gq0_prod.ancestry_preds.tsv'
system2('gsutil',args = c('cp',ancestry_data_path,'./ancestry.tsv'))
ancestry_df = read_tsv('ancestry.tsv', col_types='ic-c-') %>% rename(person_id=research_id, ancestry = ancestry_pred)
ancestry_df$ancestry = toupper(ancestry_df$ancestry)

demographics_df <- read_csv_from_bucket(name_of_file_in_bucket = 'demographics_table.csv') %>% rename(age_group_at_cdr=age_group)

################################# physical measurements specific functions #########################################
#physical_measurement_table <- read_csv_from_bucket(name_of_file_in_bucket = 'physical_measurement_table.csv')
continuous_data_summary <- function(df_measurement){
    
    value_column = colnames(select(df_measurement, -c('person_id')))
    cat(paste0('Histogram and Descriptive Statistics of ', value_column))
    
    #df_measurement <- drop_na(df_measurement)
    stats_df <- df_measurement %>%
            dplyr::summarise('Mean' = mean(df_measurement[[value_column]], na.rm = TRUE)
                             ,'Min' = min(df_measurement[[value_column]], na.rm = TRUE)
                             ,'Max' = max(df_measurement[[value_column]], na.rm = TRUE)
                             ,'1% Quantile' = quantile(df_measurement[[value_column]], 0.01, na.rm = TRUE)
                             ,'2% Quantile' = quantile(df_measurement[[value_column]], 0.02, na.rm = TRUE)
                             ,'25th Quantile' = quantile(df_measurement[[value_column]], 0.25, na.rm = TRUE)
                             ,'50th Quantile (Median)' = quantile(df_measurement[[value_column]], 0.5, na.rm = TRUE)
                             ,'75th Quantile' = quantile(df_measurement[[value_column]], 0.75, na.rm = TRUE)
                             ,'98th Quantile' = quantile(df_measurement[[value_column]], 0.98, na.rm = TRUE)
                             ,'99th Quantile' = quantile(df_measurement[[value_column]], 0.99, na.rm = TRUE)
                             ,'Standard Deviation' = sd(df_measurement[[value_column]], na.rm = TRUE)
                            )
    options(repr.plot.width = 8, repr.plot.height = 6)
    hist(x = df_measurement[[value_column]], main= str_glue('Histogram of {value_column}'), xlab= "") #paste0(value_column, " values")
    
    df_plot <- df_measurement
    boxplot(df_plot[[value_column]], main = str_glue('Boxplot of {value_column}'), xlab="")  
    View(stats_df)
    
    return(stats_df)
}

wrangle_cont_data <- function(data, demog_vars, num_col = 'name'){
    remove_cols <- append('person_id',demog_vars)
    pivot_cols <- unlist(colnames(data))
    pivot_cols <- pivot_cols[pivot_cols %in% remove_cols == FALSE]
    
    
    data_long <- pivot_longer(data, cols = all_of(pivot_cols))
    data_long <- subset(data_long[!is.na(data_long$value),], select = -c(value))
    
    groupby_cols = unlist(append(demog_vars, 'name'))
    data_long_count <- data_long%>%group_by(across(all_of(groupby_cols)))%>%summarize(n_pids = n_distinct(person_id))
    
    if (num_col == 'overall'){ #For overall counts
        num <- n_distinct(data$person_id)
        data_long_count$numerator <- num}
    else{
        num_df <- data_long[c('person_id',num_col)]%>%group_by(across(all_of(num_col)))%>%
                  summarize(numerator = n_distinct(person_id))
        data_long_count <- left_join(data_long_count, num_df, by = num_col)
        }
    
    return(data_long_count)
    }

demographic_data_summary <- function(df_measurement){
    
    # ~~~Demographics~~~
    # Age
    age_pm_long_count<- wrangle_cont_data(data = inner_join(df_measurement, demographics_df[c('person_id','age_group_at_cdr')]
                                                            , by = 'person_id'), demog_var = 'age_group_at_cdr')
    View(barplot(df = age_pm_long_count, X = `age_group_at_cdr`, n_col = 'n_pids'
                 , plot_title = '\n\n~~~Demographics~~~\n\n\n\nAge at CDR')+ guides(fill="none"))

    # Sex at Birth
    sex_pm_long_count<- wrangle_cont_data(data = inner_join(df_measurement, demographics_df[c('person_id','sex_at_birth')]
                                                          , by = 'person_id'), demog_vars = 'sex_at_birth')
    View(barplot(df = sex_pm_long_count, X = `sex_at_birth`, n_col = 'n_pids'
                 , plot_title = 'Sex at Birth', h = 5)+ guides(fill="none"))


    # Ancestry
    ancestry_pm_long_count<- wrangle_cont_data(data = inner_join(df_measurement, ancestry_df[c('person_id','ancestry')]
                                                                 , by = 'person_id'), demog_vars = 'ancestry')
    View(barplot(df = ancestry_pm_long_count, X = `ancestry`, n_col = 'n_pids'
                 , plot_title = 'Ancestry\n', h = 5)+ guides(fill="none"))


    # Sex at birth and Ancestry
    ancestry_sex_m <- inner_join(df_measurement, demographics_df[c('person_id','sex_at_birth')], by = 'person_id')
    ancestry_sex_m <- inner_join(ancestry_sex_m, ancestry_df[c('person_id','ancestry')], by = 'person_id')
    ancestry_sex_m_long_count<- wrangle_cont_data(data = ancestry_sex_m, demog_vars = list('sex_at_birth','ancestry')
                                                   , num_col = 'ancestry')

    View(barplot(df = ancestry_sex_m_long_count, X = `sex_at_birth`, n_col = 'n_pids', Facet = "ancestry"
                , plot_title = 'Sex at Birth & Ancestry\n\n'
                , facet_nrow = 4, facet_col = 2, h = 10))

    # Age at CDR and Ancestry
    ancestry_age_m <- inner_join(df_measurement, demographics_df[c('person_id','age_group_at_cdr')], by = 'person_id')
    ancestry_age_m <- inner_join(ancestry_age_m, ancestry_df[c('person_id','ancestry')], by = 'person_id')
    ancestry_age_m_long_count<- wrangle_cont_data(data = ancestry_age_m, demog_vars = list('age_group_at_cdr','ancestry')
                                                   , num_col = 'ancestry')
    View(barplot(df = ancestry_age_m_long_count, X = `age_group_at_cdr`, n_col = 'n_pids', Facet = "ancestry"
                , plot_title = 'Age at CDR & Ancestry\n\n'
                , facet_nrow = 4, facet_col = 2, h = 15))
     
 }

pm_data_summary <- function(concept_id){
    df_measurement = read_csv_cols_from_bucket(name_of_file_in_bucket = 'physical_measurement_table.csv'
                                                , concept_id = concept_id)
    title = str_to_upper(gsub('-',' ', concept_id))
     
    cat(str_glue('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ {title} Summary ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'))
    cat("\n\n")
    continuous_data_summary(df_measurement)
    
    # ~~~Demographics~~~
    demographic_data_summary(df_measurement)
     
 }
     
