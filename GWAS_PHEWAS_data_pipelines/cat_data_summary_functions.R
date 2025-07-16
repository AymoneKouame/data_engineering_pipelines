# R code to extract, analyze and sumarize, including visualizations, all categorical datasets (case/controls).

##################################################### SET UP ##############################################################
package_list <- c('bigrquery','tidyverse','dplyr','janitor', 'data.table')
for (pkg in package_list) if(!pkg %in% installed.packages()) {install.packages(pkg, quiet = T)}

library(bigrquery, warn.conflicts = F, quietly = T)
library(tidyverse, warn.conflicts = F, quietly = T)
library(janitor, warn.conflicts = F, quietly = T)
library(dplyr, warn.conflicts = F, quietly = T)
library(data.table)
options(dplyr.summarise.inform = FALSE)

# the current workspace's dataset variable (to use in the query)
dataset <- Sys.getenv('WORKSPACE_CDR')
billing_project <- Sys.getenv('GOOGLE_PROJECT')
my_bucket <- Sys.getenv('WORKSPACE_BUCKET')

# helper function to make the download easier
download_data <- function(query) {
    tb <- bq_project_query(Sys.getenv('GOOGLE_PROJECT'), page_size = 25000,
                           query = query, default_dataset = Sys.getenv('WORKSPACE_CDR'))
    bq_table_download(tb)
}


read_csv_cols_from_bucket <- function(directory = 'notebooks/all_x_all/', name_of_file_in_bucket, concept_id, remove_dot = 'no'){
    
    #reads person_id and concept id columns in a df from the input csv file in teh bucket
    #if input csv file is already read from the bucket to the env, it simply reads the person_id and concept id columns in a df
    my_bucket = Sys.getenv('WORKSPACE_BUCKET')
    
    if (file.exists(name_of_file_in_bucket)){my_dataframe  <- fread(name_of_file_in_bucket
                                                                  , select = c("person_id", str_glue("{concept_id}")))
    } else {
            if (remove_dot == 'no'){
                    system(paste0("gsutil cp ", my_bucket, "/", directory, name_of_file_in_bucket, " ."), intern=T)              
            } else {system(paste0("gsutil cp ", my_bucket, "/", directory, name_of_file_in_bucket), intern=T)}
        
            my_dataframe  <- fread(name_of_file_in_bucket, select = c("person_id", str_glue("{concept_id}")))
            }   
    my_dataframe  <- as.data.frame(my_dataframe)
    return(my_dataframe)
    }

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


barplot <- function(df, plot_title, n_col, X, Y = `Percentage`
                             , Fill = "Cases", Facet = "Cases"
                             , fill_palette = "Pastel1"
                             , facet_nrow = 1, facet_col = 2, base_text_size =9, w = 14, h = 5){
    
    df["Percentage"] <- round((df[n_col]/df['numerator'])*100,2)
    df["Label"] <- paste0(format(df[[n_col]], big.mark=","), ' (', df$Percentage, '%)')    
    #df$Legend <- factor(paste0('\n',df[[Fill]], ': N=', format(df$numerator, big.mark=","), ' (100%)'))
    if (!is.null(Facet)){df["Facet"] <- paste0('\n\n',df[[Facet]], '\nN=', format(df$numerator, big.mark=","), ' (100%)')}
    ###### Plot #####
    #X_string = deparse(substitute(X)); h*(n_distinct(df[X_string])
    options(repr.plot.width = w, repr.plot.height = h) #~1.5 per bar
    p <- ggplot(data=df, aes(x={{X}}, y={{Y}}, fill = {{Fill}}
                            )) +
            geom_bar(stat="identity", position = 'dodge2') +
            geom_text(aes(label= Label), hjust="inward", vjust = 0.5, size=(base_text_size/3)+1
                      , position = position_dodge(width = 0.9)) +
            labs(x = '', y = '', title = plot_title) +
            theme_minimal()+ 
            #scale_fill_grey(start = 0.7, end = 0.9) +
            scale_fill_brewer(palette = fill_palette) +
            #scale_fill_brewer()+
            theme(axis.text.x = element_blank(), axis.text.y = element_text(size = base_text_size+4)
                  , legend.title = element_blank()
                  , legend.text = element_text(size = base_text_size+5)
                  , legend.position = "top", legend.box = "horizontal"
                  , plot.title = element_text(hjust = 0.5)
                  , title = element_text(size = base_text_size+4)) +
            #scale_fill_mn(labels = c(new_legend)) +
            coord_flip()

    
    if (!is.null(Facet)){
        p <- p+facet_wrap(~Facet, nrow = facet_nrow, ncol = facet_col)+
                theme(strip.text.x = element_text(size = base_text_size+6))+theme(legend.position = "none")
            }
    return(p)
    }

############################ Phecode AND Survey data specific (cat dat with TRUE/FALSE (CASE/CONTROLS))

wrangle_cat_data <- function(data_df){
    colnames(data_df) = c('person_id', 'Cases')
    data_df$Cases[data_df$Cases ==TRUE]<-'Cases'
    data_df$Cases[data_df$Cases ==FALSE]<-'Controls'
    #phecode_df$Cases[is.na(phecode_df$Cases)]<- 'NA'    
    data_df <- data_df[!is.na(data_df$Cases),] # NEW removing NAs
    return(data_df)
    }

count_by <- function(data_df, var, var_df){    

    var_df = unique(var_df[c('person_id',var)])    
    merged_df = inner_join(data_df, var_df, by = 'person_id') 
    counts_df <- merged_df %>% dplyr::group_by(merged_df[var], Cases) %>% 
                    dplyr::summarise('n_pids' = n_distinct(person_id))
    
    counts_df <- cbind(counts_df, numerator=NA)
    counts_df$numerator[counts_df$Cases =='Cases'] <- n_distinct(merged_df$person_id[merged_df$Cases =='Cases'])
    counts_df$numerator[counts_df$Cases =='Controls'] <- n_distinct(merged_df$person_id[merged_df$Cases =='Controls'])
    counts_df$numerator[counts_df$Cases =='NA'] <- n_distinct(merged_df$person_id[merged_df$Cases =='NA'])
    
    return(counts_df)
    }

categorical_data_summary <- function(concept_id, name_of_file_in_bucket = 'pfhh_survey.csv'
                                     , datatype = 'Survey', map_concept_name = TRUE){
    # Function is ffor phecode, drug and survey data only (not measurements or physical measurements)
    
    system2('gsutil',args = c('cp','gs://fc-aou-preprod-datasets-controlled/v7/wgs/without_ext_aian_prod/vds/aux/ancestry/delta_v1_gt_no_ext_aian_gq0_prod.ancestry_preds.tsv','./ancestry.tsv'))        
    # LOAD Data
    
    ## Demographics Data
    demographics_df <- read_csv_from_bucket(name_of_file_in_bucket = 'demographics_table.csv')
    ## Ancestry Data
    ancestry_df = read_tsv('ancestry.tsv', col_types='ic-c-') %>% rename(person_id=research_id)
    ancestry_df$ancestry_pred = toupper(ancestry_df$ancestry_pred)
    ## Survey, drug or phecode Data
    #concept_filename = str_glue('{base_filename}_{concept_id}.csv')
    data_df = read_csv_cols_from_bucket(name_of_file_in_bucket = name_of_file_in_bucket, concept_id = concept_id)

    # TRANSFORM DATA
    Data_df <- wrangle_cat_data(data_df)
    age_count_df <- count_by(Data_df, var= 'age_group', var_df = demographics_df)
    
    sex_count_df <- count_by(Data_df, var= 'sex_at_birth', var_df = demographics_df)
    ancestry_count_df <- count_by(Data_df, var= 'ancestry_pred', var_df = ancestry_df)
    sex_and_ancestry_counts_df <- count_by(Data_df, var= c('ancestry_pred','sex_at_birth')
                                           , var_df = inner_join(ancestry_df[c('person_id','ancestry_pred')]
                                                                 , demographics_df[c('person_id','sex_at_birth')]
                                                                 , by = 'person_id'))

    age_and_ancestry_counts_df <- count_by(Data_df, var= c('ancestry_pred','age_group')
                                           , var_df = inner_join(ancestry_df[c('person_id','ancestry_pred')]
                                                                 , demographics_df[c('person_id','age_group')]
                                                                 , by = 'person_id'))

    
    ############################################################################################
    if (map_concept_name == TRUE){
        Map <- download_data(str_glue("SELECT concept_name FROM `{dataset}.concept` \
                                        WHERE concept_id = {concept_id} or concept_code = '{concept_id}'"))
        if (nrow(Map) == 0){concept_name = ''} 
        else {concept_name <- Map$concept_name; concept_name <- str_glue(' - {concept_name}')}
      } else {concept_name = ''}   
    concept_id = str_glue("{concept_id}{concept_name}")
    datatype = toupper(datatype)
    
    if (tolower(datatype) == 'pfhh'){
        concept_id = str_glue('{concept_id}\n(NB: Cases = Participants Who Reported Personally Having This Condition)')}
    else {concept_id = concept_id}
    
    ##################################################PLOT SUMMARIES#######################################################
    #'OVERALL SUMMARIES BY AGE, SEX AT BIRTH AND ANCESTRY
    n = 1
    View(barplot(age_count_df, n_col = "n_pids", X = `age_group`, h = 6
                , plot_title= str_glue('{datatype} {concept_id}:\n\n\n\nOVERALL SUMMARIES BY AGE, SEX AT BIRTH AND ANCESTRY\n\n\nFigure {n}: Age at CDR')))
    n = n+1
    View(barplot(sex_count_df, n_col = "n_pids", X = `sex_at_birth`, plot_title= str_glue('\nFigure {n}: Sex at Birth')
                    , fill_palette = "Blues"))#+theme(legend.position = "none")
    n = n+1
    View(barplot(ancestry_count_df, n_col = "n_pids", X = `ancestry_pred`, plot_title= str_glue('\nFigure {n}: Ancestry')
                     , fill_palette = "Greens"))
    
    
    # DETAILED SUMMARIES BY SEX AT BIRTH AND ANCESTRY
    sex_at_births = unique(sex_and_ancestry_counts_df$sex_at_birth)
    n = n+1
    var_sex1 = sex_at_births[1]
    df_sex1 = sex_and_ancestry_counts_df[sex_and_ancestry_counts_df$sex_at_birth == var_sex1,]
    View(barplot(df_sex1, n_col = "n_pids", X = `ancestry_pred`, Fill = `sex_at_birth`, fill_palette = "Purples"
                     , plot_title= str_glue('DETAILED SUMMARIES BY SEX AT BIRTH AND ANCESTRY\n\n\nFigure {n}: {var_sex1} (Sex at Birth) by Ancestry')
                    , facet_nrow = 1, facet_col = 2))   
    for (var in sex_at_births[-1]){
        n = n+1
        df_var = sex_and_ancestry_counts_df[sex_and_ancestry_counts_df$sex_at_birth == var,]
        View(barplot(df_var, n_col = "n_pids", X = `ancestry_pred`, Fill = `sex_at_birth`, fill_palette = "Purples"
                     , plot_title= str_glue('\nFigure {n}: {var} by Ancestry')
                    , facet_nrow = 1, facet_col = 2))
        }

    
    # DETAILED SUMMARIES BY AGE AND ANCESTRY
    ages = unique(age_and_ancestry_counts_df$age_group)
    n = n+1
    var_age1 = ages[1]
    df_age1 = age_and_ancestry_counts_df[age_and_ancestry_counts_df$age_group == var_age1,]
    View(barplot(df_age1, n_col = "n_pids", X = `ancestry_pred`, Fill = `age_group`, fill_palette = "Pastel2"
                    , plot_title= str_glue('DETAILED SUMMARIES BY AGE AT CDR AND ANCESTRY\n\n\nFigure {n}: {var_age1} Years Old by Ancestry')
                        , facet_nrow = 1, facet_col = 2)) 
    for (var in ages[-1]){
        n = n+1
        df_var = age_and_ancestry_counts_df[age_and_ancestry_counts_df$age_group == var,]
        View(barplot(df_var, n_col = "n_pids", X = `ancestry_pred`, Fill = `age_group`, fill_palette = "Pastel2"
                         , plot_title= str_glue('\nFigure {n}: {var} Years Old by Ancestry')
                              , facet_nrow = 1, facet_col = 2))
        }
    }