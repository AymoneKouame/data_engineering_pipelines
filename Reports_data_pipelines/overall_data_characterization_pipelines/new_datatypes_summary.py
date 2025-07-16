import pandas as pd
from utilities import utilities as u

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