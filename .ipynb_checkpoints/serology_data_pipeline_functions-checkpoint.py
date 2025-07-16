from datetime import datetime
import pandas as pd
import numpy as np
############ Prelim Functions

### To save files with today's date
def save_this_csv_or_xl(df, desired_filename, my_directory = data_directory, filetype = 'csv', timestamp = 'today',
                  indx = False):
    print(colored('''File will save with the default time stamp `today`. 
    To Change this chose timestamp = 'today.time.seconds', 'today.time', or 'time' ''', 'magenta'))
    
    print(colored('''\nFile will save as .csv by default. 
    To Change that, chose `filetype = 'xlsx' ''', 'magenta'))
    
    print(colored('''\n `index = False` by default. You can choose `index = 'True' ''', 'white'))
    
    today = str(datetime.today().strftime('%Y-%m-%d'))
    time = str(datetime.now().time())[:5]
    time_second = str(datetime.now().time())[:8]
    today_time = (today+'_'+time.replace(':','.'))
    now = (today+'_'+time_second).replace(':','.')
    
    if timestamp == 'today.time.seconds':
        filename_tp = desired_filename+'_'+now
        
    else:
        if timestamp == 'today.time':
            filename_tp = desired_filename+'_'+today_time
            
        else:
            if timestamp == 'today':
                 filename_tp = desired_filename+'_'+today
                    
    if filetype == 'csv':
        df.to_csv(my_directory+filename_tp+'.csv', index = indx)
        
    else:
        if filetype == 'xlsx':
            df.to_excel(my_directory+filename_tp+'.xlsx', index = indx)
        

### Function to transform manifest files received from biobank
def transform_manifest(df, currend_bid_fieldname, master_list_df = None, 
                       negative_controls_master_df = None, which_df = None):
    
    dsample = df[df[currend_bid_fieldname].str.startswith('A')]
    dsample['biobank_id'] = [i.split('A')[1] for i in dsample[currend_bid_fieldname]]
    dsample['biobank_id'] = dsample['biobank_id'].astype('int64')
    dsample = dsample.drop(currend_bid_fieldname, axis = 1)

    if which_df == None:
        print(colored("which_df default value is None.\nPlease set it to these possible values: 'positive controls', 'negative controls'  or 'test samples' ", 'red'))    

    else:
        if which_df == 'positive controls':
            ## POSITIVE CONTROLS
            d_pos = df[~df[currend_bid_fieldname].str.startswith('A')]
            d_pos['Positive Control ID'] = d_pos[currend_bid_fieldname]
            DF = d_pos.drop(currend_bid_fieldname, axis = 1)
            DF['Positive Control'] = 'Yes'
            DF['Negative Control'] = 'No'
            print('positive controls nunique(): '+str(DF['Positive Control ID'].nunique()))
            #display(DF.head(5))
            
        else:
            if which_df == 'negative controls':
                negative_controls = negative_controls_master_df

                # NEGATIVE CONTROLS
                DF = pd.merge(dsample, negative_controls)
                print('negative controls nunique(): '+str(DF.biobank_id.nunique()))
                #display(DF.head(5))

            else:
                if which_df == 'test samples':
                    noncontrols_df = master_list_df
                    #TEST SAMPLES
                    DF = pd.merge(dsample,noncontrols_df)
                    print('test samples nunique(): '+str(DF.biobank_id.nunique()))
                    #display(DF.head(5))
                    
                else:
                    print("Error: please check your spelling for 'which_df'")
                    

    return DF

### Function to concatenate neg conrols, postive controls and test samples in one file with their respective demogaphics
## Uses transform_manifest() and save_this_csv()
def pull_demographics(df, currend_bid_fieldname, save_as_filename):
    
    which_df = ['test samples','negative controls','positive controls']
    
    # TEST SAMPLE DEMOGRAPHICS
    print(colored(which_df[0], 'magenta'))
    DF_sample = transform_manifest(df, currend_bid_fieldname, which_df = which_df[0])
    
    # NEGATIVE CONTROL DEMOGRAPHICS
    print(colored(which_df[1], 'magenta'))
    DF_neg = transform_manifest(df, currend_bid_fieldname, which_df= which_df[1])
    
    # POSITIVE CONTROLS
    print(colored(which_df[2], 'magenta'))
    DF_pos = transform_manifest(df, currend_bid_fieldname, which_df[2])
    display(DF_pos.head(3))
    
    # PUTTING AL TOGETHER
    DF_demograhics = pd.concat([DF_sample, DF_neg], sort = False)
    DF_demograhics_all = pd.concat([DF_demograhics, DF_pos], sort = False)
    DF_demograhics_all['Negative Control'] = DF_demograhics_all['Negative Control'].fillna('No')
    DF_demograhics_all['Positive Control'] = DF_demograhics_all['Positive Control'].fillna('No')
    
    save_this_csv(DF_demograhics_all, save_as_filename, data_directoty)
    save_this_csv(DF_demograhics_all, save_as_filename, david_directory)
    
    print(colored('Done and Saved!', 'magenta'))
    display(DF_demograhics_all.head())

    
    return DF_demograhics_all

############ Functions to perform each step of the plating strategy

### function to pull batch of desired size from master sample list
def pull_batch(master_ls, batch_size, batch_number, previous_batch = None):  
    '''Sorts sample master file from most recent to oldest and pulls batch of first n (n = desired batch size).
       When applicable, excludes list of biobank_ids in the previous batch from new batch
       '''
    
    DF = master_ls[['participant_id','biobank_id','DateBloodSampleCollected', 'DateBloodSampleReceived', 'state', 'Negative Control', 'Positive Control']].drop_duplicates()
    
    # If there is a previous batch,
    # keep only pids that are in the master list and not in the previous batch
    # Else, proceed
    if previous_batch.empty:
        DF = DF
        
    else:
        keep = pd.DataFrame(set(master_ls.biobank_id) - set(previous_batch.biobank_id)).rename(columns = {0:'biobank_id'}) 
        DF = pd.merge(DF, keep, how = 'inner')
    
    # Sort Date of collection from most recent to oldest in the master file
    # then select the first n(batch size) rows
    new_batch = DF.sort_values('DateBloodSampleCollected', ascending= False).iloc[:batch_size,:]
    
    print(colored('Batch #' +str(batch_number)+' of ' +str(new_batch.biobank_id.nunique()) + ' participants pulled from Master List, sorted in descending order of collection dates, is ready.', 'blue'))
    return new_batch

### Functions to randomize entire batch by State and add location columns to batch
def randomize_and_locate(batch_df, batch_size, rand_state):
    '''Function to add bay,freezer and rack location columns to pulled batch and randomize by state entire batch 
        to get it ready for optimization'''
    
    # randomize by state = rnadomly shuffle states
    batch_n_state = batch_df[['state']].sample(n = batch_size, random_state = rand_state)
    batch_n_randomState = batch_df.sample(n = batch_size, random_state= rand_state) 
    
    print('Shape check:' + str(batch_n_randomState.shape))
 
    return batch_n_randomState

def randomize_state(batch_df, batch_size, rand_state):
    '''Function to add bay,freezer and rack location columns to pulled batch and randomize by state entire batch 
        to get it ready for optimization'''
    
    # randomize by state = rnadomly shuffle states
    batch_n_state = batch_df[['state']].sample(n = batch_size, random_state = rand_state)
    batch_n_randomState = batch_df.sample(n = batch_size, random_state= rand_state) 
    
    print('Shape check:' + str(batch_n_randomState.shape))
 
    return batch_n_randomState

### Function to optimize the Sequence and Group participants by plates of n size
def optimize_sequence(n_wells, df):  #df = = samples_groups
    '''Function to Create Optimized Sequence 
       Groups participants in plates of n_wells participants
       n_wells is the number of wells/the number of people per plate '''

    # create empty 'plate' column
    df['plate'] = int()
    
    # SEQUENCE OPTIMIZATION:
     ## Order the batch by Description (Bay, Freezer), Sequence and Rack (optimum sequence order provided by biobank)
     ## Then assign plate numbers to each participant, starting with plate #1
     ## with n_wells participants per plate 
    n = 0
    plate_number = 0

    for pid in df.sort_values(['Description', 'Sequence','Rack']).biobank_id: 
        pids_per_well = df.sort_values(['Description', 'Sequence', 'Rack'])[['biobank_id']][n:n_wells+n]
        ind = pids_per_well.index.values
        plate_number += 1

        for i in ind:
            df.loc[i, 'plate'] = plate_number

        n += n_wells
        
    display(df[['participant_id','plate']].rename(columns = {'participant_id':'pids_per_plate'}).groupby('plate').count())
    print(colored('Done! We have ' +str(n_wells)+' wells per plate, with ' +str(df.plate.nunique()) + ' plates in total. The last few plates may have different numbers', 'blue'))
    
    return  df

### Function to Group participants by plates of n size
def plate_em(n_wells, df, rand_state, batch_size):  #df = = samples_groups
    '''Function to Create Optimized Sequence 
       Groups participants in plates of n_wells participants
       n_wells is the number of wells/the number of people per plate '''

    # create empty 'plate' column
    df['plate'] = int()
    
    # SEQUENCE OPTIMIZATION:
     ## Randomly shuffles Pids
     ## Then assign plate numbers to each participant, starting with plate #1
     ## with n_wells participants per plate 
    
    df = df.sample(n = batch_size, random_state = rand_state)
    
    n = 0
    plate_number = 0
    for pid in df.biobank_id: 
        pids_per_well = df[['biobank_id']][n:n_wells+n]
        ind = pids_per_well.index.values
        plate_number += 1

        for i in ind:
            df.loc[i, 'plate'] = plate_number

        n += n_wells
    
    df = df.drop('DateBloodSampleReceived', axis = 1)
    display(df[['participant_id','plate']].rename(columns = {'participant_id':'pids_per_plate'}).groupby('plate').count())
    print(colored('Done! We have ' +str(n_wells)+' wells per plate, with ' +str(df.plate.nunique()) + ' plates in total. The last few plates may have different numbers', 'blue'))
    
    return  df

### Function to add negative controls, including repeated pids
def batch_negative_controls(plated_samples_df, batch_df, negControls_master, unrepeated_negControls, 
                          number_plates, rand_state, previous_batch_negControls = None, duplicate = 'Yes'):
    ''' This funtions select batch negative controls, then appends a repeated list of pids 
        Then merges with the samples from 2020. This will be the input for the optimization function'''
    
    # 1 Remove previous batches negative controls from master list, if applicable
    if previous_batch_negControls.empty:
        negControls_master = negControls_master
        
    else:
        keep = pd.DataFrame(set(negControls_master.biobank_id) - set(previous_batch_negControls.biobank_id)).rename(columns = {0:'biobank_id'}) 
        negControls_master = pd.merge(negControls_master, keep)
        
    print(colored('How many negative controls are left in the master list \nafter removing the negative controls from previous batches? : '+str(negControls_master.biobank_id.nunique()), 'red'))

    
    # 2 MATCH  negative controls on state
    neg_controls_df= pd.merge(negControls_master[['biobank_id', 'participant_id','state', 'DateBloodSampleCollected', 'Negative Control', 'Positive Control']].drop_duplicates(), 
                                batch_df[['state']].drop_duplicates()).drop_duplicates()#.reset_index()
    
    neg_controls_df = neg_controls_df#master_neg_state_matched
    print(colored('\nHow many negative controls are left in the master list \nafter removing the negative controls from previous batches and matching on state? : '+str(neg_controls_df.biobank_id.nunique()), 'red'))
    print('Any state in the matched df that is not in the batch?:')
    display(set(neg_controls_df.state) - set(batch_df.state))
    
    print(colored('\nQC State Matching alogirthm: ', 'green'))
    print('\nN Unique states in neg control master list state-matched vs batch_df states: '+str(neg_controls_df.state.nunique()) +', '+ str(batch_df.state.nunique()))
    print('Unique states in neg control master list state-matched vs batch_df states: ')
    display(neg_controls_df[['state']].drop_duplicates().sort_values('state'), batch_df[['state']].drop_duplicates().sort_values('state'))

    ## 3 randomly select the batch negative controls
    if duplicate == 'Yes':
        n_total_controls = math.floor(number_plates/2)
    else:
        if duplicate == 'No':
            n_total_controls = number_plates

    batch_neg_controls_df = neg_controls_df.sample(n = int(n_total_controls), 
                                                random_state = rand_state).reset_index().drop('index', axis = 1)
 
    batch_neg_controls_df = batch_neg_controls_df.merge(negControls_master[['biobank_id', 'participant_id','state', 'DateBloodSampleCollected', 'Negative Control', 'Positive Control']].drop_duplicates()).drop_duplicates()

    batch_neg_controls_df = batch_neg_controls_df.sort_values('biobank_id', ascending = True)
    batch_neg_controls_df2 = batch_neg_controls_df.sort_values('biobank_id', ascending = False) # for duplication
    batch_neg_controls_df = pd.concat([batch_neg_controls_df,batch_neg_controls_df2])
    
    

    ### 4 Assign negative controls to plates, 1 per plate
    batch_neg_controls_df['plate']= int()
    batch_neg_controls_df['plate']= range(1,int(number_plates)+1)
    
    print(colored('QC State Matching alogirthm 2: ', 'green'))
    
    print('\nN Unique states in batch neg controls vs batch_df states: '+str(batch_neg_controls_df.state.nunique()) +', '+ str(batch_df.state.nunique()))
    print('Unique states in batch neg controls vs batch_df states: ')
    display(batch_neg_controls_df[['state']].drop_duplicates().sort_values('state'), batch_df[['state']].drop_duplicates().sort_values('state'))

    print(colored('Other QCs: ', 'green'))
    print('unique pids:' + str(batch_neg_controls_df.biobank_id.nunique()))
    print('count of pids:' + str(batch_neg_controls_df.biobank_id.count()))
    print('number of plates (unique and count):' + str(batch_neg_controls_df.plate.nunique())+' and ' + str(batch_neg_controls_df.plate.count()))
    print('shape:' + str(batch_neg_controls_df.shape))
    
    #batch_neg_controls_df = batch_neg_controls_df.drop('state', axis = 1)
    
    return batch_neg_controls_df

### Function to add positive controls
def batch_positive_controls(pos_controls_df, number_plates, rand_state, aliquots_needed):
    '''Function to get positive controls '''

    # choose n samples from positive controls, randomly from high/med/low
    pos_controls = pos_controls_df.drop_duplicates()
     
    pos_controls = pos_controls.sample(n = int(number_plates/2), random_state= rand_state)

    ## QC - chheck volume
    check = pd.merge(pos_controls, pos_controls_df.drop_duplicates())
    print('checking batch positive control volume per sample:')
    display(check.iloc[:,1].sum()/aliquots_needed)

    pos_controls = pos_controls.sort_values('Sample ID', ascending = True)
    pos_controls2 = pos_controls.sort_values('Sample ID', ascending = False) # for duplication
    pos_controls_DF = pd.concat([pos_controls,pos_controls2])

    
    ### Assign POS controls to plates
    pos_controls_DF['plate']= int()
    pos_controls_DF['plate']= range(1,int(number_plates)+1)
    
    pos_controls_DF['Positive Control']= 'Yes'
    pos_controls_DF['Negative Control']= 'No'
 
    return pos_controls_DF

### Function to get final deliverable
def final_deliverable(optimized_seq, neg_controls, pos_controls, bid_fieldname):
    '''Function to add all together and return final deliverable'''

    Serology_final_dataset = pd.concat([optimized_seq, neg_controls], sort=True)
    Serology_final_dataset = pd.concat([Serology_final_dataset, pos_controls], sort=True)
    
    Serology_final_dataset = Serology_final_dataset.rename(columns = {'Sample ID':'Positive Control ID'})
    
    print(colored('QC Number of positive, negative and test samples before sending final file: ', 'white'))
    display(Serology_final_dataset.head())

    negs = Serology_final_dataset[Serology_final_dataset['Negative Control'] == 'Yes']
    n_negs = negs[bid_fieldname].nunique()
    testSamples =Serology_final_dataset[(Serology_final_dataset['Negative Control'] == 'No') &
                                        (Serology_final_dataset['Positive Control'] == 'No')]
    n_testSamples = testSamples[bid_fieldname].nunique()
    
    if 'Positive Control ID' in Serology_final_dataset.columns:
        n_pos = Serology_final_dataset[Serology_final_dataset['Positive Control'] == 'Yes']['Positive Control ID'].nunique()
        total_sample = n_pos+n_negs+n_testSamples

    else:
        n_pos = Serology_final_dataset[Serology_final_dataset['Positive Control'] == 'Yes'][bid_fieldname].nunique()
        total_sample = Serology_final_dataset[bid_fieldname].nunique()
    
    print(colored('Total Unique Non Control Samples: ' + str(n_testSamples), 'magenta')) 
    print(colored('Total Unique Negative Controls: ' + str(n_negs), 'magenta')) 
    print(colored('Total Unique Positive Controls: ' + str(n_pos), 'magenta')) 
    print(colored('Total Unique Samples Tested (including controls): ' + str(total_sample), 'blue')) 
    
    print(colored('Date Blood Sample Collected Max: ' + str(testSamples.DateBloodSampleCollected.max()), 'green')) 
    print(colored('Date Blood Sample Collected Min: ' + str(testSamples.DateBloodSampleCollected.min()), 'green'))
    print(colored('Date Negative Control Blood Sample Collected Max: ' + str(negs.DateBloodSampleCollected.max()), 'green')) 
    print(colored('Date Negative Control Blood Sample Collected Min: ' + str(negs.DateBloodSampleCollected.min()), 'green'))
   
    Serology_final_dataset[['biobank_id','plate', 'Positive Control ID']].drop_duplicates().sort_values('plate').groupby('plate').nunique()
    
    return Serology_final_dataset

########## Quality Control

### Function to Check the distributions of datasets by a specified variable
def get_dist(DF, dist_var, group = None, groupnumber = None):
    '''group determines whether the distribution is to be done by plate or on the entire dataset.
       dist_var is the variable for which we check the distribution -- ie race, state 
       if group is 'plate', then, group number is the plate number'''
            
    if group == None:
        df = pd.DataFrame(DF[['participant_id', dist_var]].drop_duplicates()[dist_var].value_counts())
        df['%ofTheGroup'] = (df[dist_var]/DF.participant_id.nunique())*100
        
    else:        
        df1 = DF[DF[group] == groupnumber] 
        df = pd.DataFrame(df1[['participant_id', dist_var]].drop_duplicates()[dist_var].value_counts())
        df['%ofTheGroup'] = (df[dist_var]/df1.participant_id.nunique())*100            
    return df

### Function to QC the numbers in the final batch/list deliverable
def QC_batch_numbers(DF, bid_fieldname):
    ''' Function to check the bumber of positive, negative and test samples before sending final file.'''
    
    DF.DateBloodSampleCollected = pd.to_datetime(DF.DateBloodSampleCollected)
    specimens = DF['Specimen ID'].nunique()
    negs = DF[DF['Negative Control'] == 'Yes']
    n_negs = negs[bid_fieldname].nunique()
    testSamples =DF[(DF['Negative Control'] == 'No') &
                                        (DF['Positive Control'] == 'No')]
    n_testSamples = testSamples[bid_fieldname].nunique()
    
    if 'Positive Control ID' in DF.columns:
        n_pos = DF[DF['Positive Control'] == 'Yes']['Positive Control ID'].nunique()
        total_sample = n_pos+n_negs+n_testSamples

    else:
        n_pos = DF[DF['Positive Control'] == 'Yes'][bid_fieldname].nunique()
        total_sample = DF[bid_fieldname].nunique()
    
    print(colored('Total Non Control Samples: ' + str(n_testSamples), 'magenta')) 
    print(colored('Total Unique Negative Controls: ' + str(n_negs), 'magenta')) 
    print(colored('Total Unique Positive Controls: ' + str(n_pos), 'magenta')) 
    print(colored('Total Unique Samples Tested (including controls): ' + str(total_sample), 'blue'))
    print(colored('Total Unique Samples/Specimen IDs(including controls): ' + str(specimens), 'blue'))
    
    print(colored('Date Blood Sample Collected Max: ' + str(testSamples.DateBloodSampleCollected.max()), 'green')) 
    print(colored('Date Blood Sample Collected Min: ' + str(testSamples.DateBloodSampleCollected.min()), 'green'))
    print(colored('Any Null Blodd Sample Dates in the non controls?: ' + str(testSamples[testSamples.DateBloodSampleCollected.isnull()][bid_fieldname].nunique()), 'green')) 
    
    print(colored('\nDate Negative Control Blood Sample Collected Max: ' + str(negs.DateBloodSampleCollected.max()), 'green')) 
    print(colored('Date Negative Control Blood Sample Collected Min: ' + str(negs.DateBloodSampleCollected.min()), 'green'))

### Function to check the bumber of positive, negative and test samples numbers
def get_specimens_counts(DF, which_field, plated = 'Yes'):
    ''' Function to check the bumber of positive, negative and test samples numbers.'''
    
    print(colored('Counts for the '+str(which_field)+' field:', 'red'))
    
    negs = DF[DF['Negative Control'] == 'Yes']
    n_negs = negs[which_field].nunique()
    
    testSamples =DF[(DF['Negative Control'] == 'No') & (DF['Positive Control'] == 'No')]
    n_testSamples = testSamples[which_field].nunique()
    
    count1 = int(testSamples[which_field].count())+ int(negs[which_field].count())
    
    pos = DF[DF['Positive Control'] == 'Yes']
    if 'Positive Control ID' in DF.columns:      
        n_pos = pos['Positive Control ID'].nunique()
        count_pos = pos['Positive Control ID'].count()
        total_sample = n_pos+n_negs+n_testSamples

    else:
        n_pos = pos[which_field].nunique()
        count_pos = pos[which_field].count()
        total_sample = DF[which_field].nunique()
    
    print(colored(' \nN Unique Non Controls: ' + str(n_testSamples), 'magenta')) 
    print(colored(' Count Non Controls: ' + str(testSamples[which_field].count()), 'magenta')) 
    print(colored('Date Non Controls Blood Sample Collected Max: ' + str(testSamples.DateBloodSampleCollected.max()), 'magenta')) 
    print(colored('Date Non Controls Blood Sample Collected Min: ' + str(testSamples.DateBloodSampleCollected.min()), 'magenta'))
    
    print(colored(' \nN Unique Negative Controls: ' + str(n_negs), 'green')) 
    print(colored(' Count Negative Controls: ' + str(negs[which_field].count()), 'green'))
    print(colored('Date Neg Controls Blood Sample Collected Max: ' + str(negs.DateBloodSampleCollected.max()), 'green')) 
    print(colored('Date Neg Controls Blood Sample Collected Min: ' + str(negs.DateBloodSampleCollected.min()), 'green'))

    
    print(colored(' \nN Unique Positive Controls: ' + str(n_pos), 'blue')) 
    print(colored(' Count Positive Controls: ' + str(count_pos), 'blue')) 
    print(colored('Date Pos Controls Blood Sample Collected Max: ' + str(pos.DateBloodSampleCollected.max()), 'blue')) 
    print(colored('Date Pos Controls Blood Sample Collected Min: ' + str(pos.DateBloodSampleCollected.min()), 'blue'))
    
    print(colored(' \nN Unique All Samples (including controls): ' + str(total_sample), 'yellow'))
    print(colored(' Count All Samples (including controls): ' + str(int(count1+count_pos)), 'yellow')) 
    
    
    if plated == 'Yes':
        print(colored(' \nN Unique Plates: ' + str(DF.plate.nunique()), 'magenta')) 
        print(colored(' Count Plates: ' + str(DF.plate.count()), 'magenta'))