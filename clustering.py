import numpy as np
import pandas as pd
import os
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from numpy import array, unique, zeros
warnings.filterwarnings("ignore")
import plotly.express as px


def get_one_hot(value, category):
    if category == "etnicity":
        categories = [ "", "West European", "North European (Nordic)", "South European", "South-East European", "East European", "Arab", "Jewish", "Turkish", 
            "Iranian and Central Asian","Other North African and Middle Eastern", "West and Central African", "Africa's Horn", "East and South African", 
            "South Asian", "Mainland and Buddhist South-East Asian", "Maritime and Muslim South-East Asian", "Chinese Asian", "North-East Asian", 
            "South American", "Central American", "English-speaking Caribbean", "Non-English-speaking Caribbean", "North American", "Australasian"]
    elif category == "etnicityMacro":
        categories =[ "","European", "North African, Middle Eastern and Central Asian", "Sub-Saharan African", "South and South-East Asian",
             "East Asian", "Latin American", "Caribbean", "North American and Australasian"]
    elif category == "migrant":
        categories = [ "No migratory background", "Child native-born, one parent foreign-born", "Child native-born, both parents foreign-born", 
            "Child foreign-born, neither parent foreign-born", "Child and one parent foreign-born", "", "Child and both parents foreign-born",""]
    else: 
        categories = []

    one_hot_encoding = zeros(len(categories), dtype=int)
    index = categories.index(value)
    if index > 0:
        one_hot_encoding[index] = 1
    return one_hot_encoding.tolist()

def get_val(data, category, type, scenario, level, session = 1):
    value = 0
    if category=="interaction":
        if session == 1:
            if len(data) > 0:
                value = data[0]["interaction"][type]["breakdown"][scenario-1]["levels"][level-1]["count"]
            else:
                value = 0
        else: 
            count = 0
            for row in data:
                if count > 0:
                    value += row["interaction"][type]["breakdown"][scenario-1]["levels"][level-1]["count"]
                count += 1
        return value
    elif category == "helps":
        if session == 1:
            if len(data) > 0:
                value = data[0]["helps"]["breakdown"][scenario-1]["levels"][level-1]["count"]
            else:
                value = 0
        else: 
            count = 0
            for row in data:
                if count > 0:
                    value += row["helps"]["breakdown"][scenario-1]["levels"][level-1]["count"]
                count += 1
        return value
    else:
        return value
    
def get_performance(data,scenario,level, session = 1):
    value = 0
    if session == 1:
        if len(data) > 0:
            scr = data[0]["scores"]["breakdown"][scenario-1]["levels"][level-1]["score"]
            max = data[0]["scores"]["breakdown"][scenario-1]["levels"][level-1]["max"]
            que = data[0]["scores"]["breakdown"][scenario-1]["levels"][level-1]["question"]
        else:
            scr,max,que = 0, 0, 0     
    else: 
        count = 0
        scr, max, que = 0, 0, 0
        for row in data:
            if count > 0:
                scr += row["scores"]["breakdown"][scenario-1]["levels"][level-1]["score"]
                max += row["scores"]["breakdown"][scenario-1]["levels"][level-1]["max"]
                que += row["scores"]["breakdown"][scenario-1]["levels"][level-1]["question"]
            count += 1
    value = scr/max if que > 0 else 0
    return value

def get_interaction_vector(intera):
    vector = []
    # Interactions with characters
    s1_sc1_l1_interactions = get_val(intera, "interaction", "character", 1, 1)
    s1_sc1_l2_interactions = get_val(intera, "interaction", "character", 1, 2)
    s1_sc1_l3_interactions = get_val(intera, "interaction", "character", 1, 3)
    s1_sc1_l4_interactions = get_val(intera, "interaction", "character", 1, 4)
    s1_sc1_interactions_total = s1_sc1_l1_interactions + s1_sc1_l2_interactions + s1_sc1_l3_interactions + s1_sc1_l4_interactions
    
    s1_sc2_l1_interactions = get_val(intera, "interaction", "character", 2, 1)
    s1_sc2_l2_interactions = get_val(intera, "interaction", "character", 2, 2)
    s1_sc2_l3_interactions = get_val(intera, "interaction", "character", 2, 3)
    s1_sc2_l4_interactions = get_val(intera, "interaction", "character", 2, 4)
    s1_sc2_interactions_total = s1_sc2_l1_interactions + s1_sc2_l2_interactions + s1_sc2_l3_interactions + s1_sc2_l4_interactions
    s2_sc1_l1_interactions = get_val(intera, "interaction", "character", 1, 1, 2)
    s2_sc1_l2_interactions = get_val(intera, "interaction", "character", 1, 2, 2)
    s2_sc1_l3_interactions = get_val(intera, "interaction", "character", 1, 3, 2)
    s2_sc1_l4_interactions = get_val(intera, "interaction", "character", 1, 4, 2)
    s2_sc1_interactions_total = s2_sc1_l1_interactions + s2_sc1_l2_interactions + s2_sc1_l3_interactions + s2_sc1_l4_interactions
    s2_sc2_l1_interactions = get_val(intera, "interaction", "character", 2, 1, 2)
    s2_sc2_l2_interactions = get_val(intera, "interaction", "character", 2, 2, 2)
    s2_sc2_l3_interactions = get_val(intera, "interaction", "character", 2, 3, 2)
    s2_sc2_l4_interactions = get_val(intera, "interaction", "character", 2, 4, 2)
    s2_sc2_interactions_total = s2_sc2_l1_interactions + s2_sc2_l2_interactions + s2_sc2_l3_interactions + s2_sc2_l4_interactions
    
    # Change Scene interactions
    s1_sc1_l1_change_scene = get_val(intera, "interaction", "change_scene", 1, 1)
    s1_sc1_l2_change_scene = get_val(intera, "interaction", "change_scene", 1, 2)
    s1_sc1_l3_change_scene = get_val(intera, "interaction", "change_scene", 1, 3)
    s1_sc1_l4_change_scene = get_val(intera, "interaction", "change_scene", 1, 4)
    s1_sc1_change_scene_total = s1_sc1_l1_change_scene + s1_sc1_l2_change_scene + s1_sc1_l3_change_scene + s1_sc1_l4_change_scene
    s1_sc2_l1_change_scene = get_val(intera, "interaction", "change_scene", 2, 1)
    s1_sc2_l2_change_scene = get_val(intera, "interaction", "change_scene", 2, 2)
    s1_sc2_l3_change_scene = get_val(intera, "interaction", "change_scene", 2, 3)
    s1_sc2_l4_change_scene = get_val(intera, "interaction", "change_scene", 2, 4)
    s1_sc2_change_scene_total = s1_sc2_l1_change_scene + s1_sc2_l2_change_scene + s1_sc2_l3_change_scene + s1_sc2_l4_change_scene
    s2_sc1_l1_change_scene = get_val(intera, "interaction", "change_scene", 1, 1, 2)
    s2_sc1_l2_change_scene = get_val(intera, "interaction", "change_scene", 1, 2, 2)
    s2_sc1_l3_change_scene = get_val(intera, "interaction", "change_scene", 1, 3, 2)
    s2_sc1_l4_change_scene = get_val(intera, "interaction", "change_scene", 1, 4, 2)
    s2_sc1_change_scene_total = s2_sc1_l1_change_scene + s2_sc1_l2_change_scene + s2_sc1_l3_change_scene + s2_sc1_l4_change_scene
    s2_sc2_l1_change_scene = get_val(intera, "interaction", "change_scene", 2, 1, 2)
    s2_sc2_l2_change_scene = get_val(intera, "interaction", "change_scene", 2, 2, 2)
    s2_sc2_l3_change_scene = get_val(intera, "interaction", "change_scene", 2, 3, 2)
    s2_sc2_l4_change_scene = get_val(intera, "interaction", "change_scene", 2, 4, 2)
    s2_sc2_change_scene_total = s2_sc2_l1_change_scene + s2_sc2_l2_change_scene + s2_sc2_l3_change_scene + s2_sc2_l4_change_scene

    # Movement interactions    
    s1_sc1_l1_movements    = get_val(intera, "interaction", "movement", 1, 1)
    s1_sc1_l2_movements    = get_val(intera, "interaction", "movement", 1, 2)
    s1_sc1_l3_movements    = get_val(intera, "interaction", "movement", 1, 3)
    s1_sc1_l4_movements    = get_val(intera, "interaction", "movement", 1, 4)
    s1_sc1_movements_total = s1_sc1_l1_movements + s1_sc1_l2_movements + s1_sc1_l3_movements + s1_sc1_l4_movements
    s1_sc2_l1_movements    = get_val(intera, "interaction", "movement", 2, 1)
    s1_sc2_l2_movements    = get_val(intera, "interaction", "movement", 2, 2)
    s1_sc2_l3_movements    = get_val(intera, "interaction", "movement", 2, 3)
    s1_sc2_l4_movements    = get_val(intera, "interaction", "movement", 2, 4)
    s1_sc2_movements_total = s1_sc2_l1_movements + s1_sc2_l2_movements + s1_sc2_l3_movements + s1_sc2_l4_movements
    s2_sc1_l1_movements    = get_val(intera, "interaction", "movement", 1, 1, 2)
    s2_sc1_l2_movements    = get_val(intera, "interaction", "movement", 1, 2, 2)
    s2_sc1_l3_movements    = get_val(intera, "interaction", "movement", 1, 3, 2)
    s2_sc1_l4_movements    = get_val(intera, "interaction", "movement", 1, 4, 2)
    s2_sc1_movements_total = s2_sc1_l1_movements + s2_sc1_l2_movements + s2_sc1_l3_movements + s2_sc1_l4_movements
    s2_sc2_l1_movements    = get_val(intera, "interaction", "movement", 2, 1, 2)
    s2_sc2_l2_movements    = get_val(intera, "interaction", "movement", 2, 2, 2)
    s2_sc2_l3_movements    = get_val(intera, "interaction", "movement", 2, 3, 2)
    s2_sc2_l4_movements    = get_val(intera, "interaction", "movement", 2, 4, 2)
    s2_sc2_movements_total = s2_sc2_l1_movements + s2_sc2_l2_movements + s2_sc2_l3_movements + s2_sc2_l4_movements

    vector.extend([
        s1_sc1_interactions_total, s1_sc1_change_scene_total, s1_sc1_movements_total,
        s1_sc1_l1_interactions, s1_sc1_l2_interactions, s1_sc1_l3_interactions, s1_sc1_l4_interactions,
        s1_sc1_l1_change_scene, s1_sc1_l2_change_scene, s1_sc1_l3_change_scene, s1_sc1_l4_change_scene, 
        s1_sc1_l1_movements, s1_sc1_l2_movements, s1_sc1_l3_movements, s1_sc1_l4_movements, 
        
        s1_sc2_interactions_total,s1_sc2_change_scene_total, s1_sc2_movements_total,
        s1_sc2_l1_interactions, s1_sc2_l2_interactions, s1_sc2_l3_interactions, s1_sc2_l4_interactions, 
        s1_sc2_l1_change_scene, s1_sc2_l2_change_scene, s1_sc2_l3_change_scene, s1_sc2_l4_change_scene,
        s1_sc2_l1_movements, s1_sc2_l2_movements, s1_sc2_l3_movements, s1_sc2_l4_movements, 

        s2_sc1_interactions_total,s2_sc1_change_scene_total,s2_sc1_movements_total,
        s2_sc1_l1_interactions, s2_sc1_l2_interactions, s2_sc1_l3_interactions, s2_sc1_l4_interactions, 
        s2_sc1_l1_change_scene, s2_sc1_l2_change_scene, s2_sc1_l3_change_scene, s2_sc1_l4_change_scene, 
        s2_sc1_l1_movements, s2_sc1_l2_movements, s2_sc1_l3_movements, s2_sc1_l4_movements, 

        s2_sc2_interactions_total,s2_sc2_change_scene_total,s2_sc2_movements_total,
        s2_sc2_l1_interactions, s2_sc2_l2_interactions, s2_sc2_l3_interactions, s2_sc2_l4_interactions, 
        s2_sc2_l1_change_scene, s2_sc2_l2_change_scene, s2_sc2_l3_change_scene, s2_sc2_l4_change_scene, 
        s2_sc2_l1_movements, s2_sc2_l2_movements, s2_sc2_l3_movements, s2_sc2_l4_movements
    ])
    
    return vector
    
def get_help_vector(intera):
    vector = []

    s1_sc1_l1_helps = get_val(intera, "helps", "", 1, 1)
    s1_sc1_l2_helps = get_val(intera, "helps", "", 1, 2)
    s1_sc1_l3_helps = get_val(intera, "helps", "", 1, 3)
    s1_sc1_l4_helps = get_val(intera, "helps", "", 1, 4)
    s1_sc1_helps_total = s1_sc1_l1_helps + s1_sc1_l2_helps + s1_sc1_l3_helps + s1_sc1_l4_helps
    s1_sc2_l1_helps = get_val(intera, "helps", "", 2, 1)
    s1_sc2_l2_helps = get_val(intera, "helps", "", 2, 2)
    s1_sc2_l3_helps = get_val(intera, "helps", "", 2, 3)
    s1_sc2_l4_helps = get_val(intera, "helps", "", 2, 4)
    s1_sc2_helps_total = s1_sc2_l1_helps + s1_sc2_l2_helps + s1_sc2_l3_helps + s1_sc2_l4_helps

    s2_sc1_l1_helps = get_val(intera, "helps", "", 1, 1, 2)
    s2_sc1_l2_helps = get_val(intera, "helps", "", 1, 2, 2)
    s2_sc1_l3_helps = get_val(intera, "helps", "", 1, 3, 2)
    s2_sc1_l4_helps = get_val(intera, "helps", "", 1, 4, 2)
    s2_sc1_helps_total = s2_sc1_l1_helps + s2_sc1_l2_helps + s2_sc1_l3_helps + s2_sc1_l4_helps
    s2_sc2_l1_helps = get_val(intera, "helps", "", 2, 1, 2)
    s2_sc2_l2_helps = get_val(intera, "helps", "", 2, 2, 2)
    s2_sc2_l3_helps = get_val(intera, "helps", "", 2, 3, 2)
    s2_sc2_l4_helps = get_val(intera, "helps", "", 2, 4, 2)
    s2_sc2_helps_total = s2_sc2_l1_helps + s2_sc2_l2_helps + s2_sc2_l3_helps + s2_sc2_l4_helps

    vector.extend([
        s1_sc1_helps_total,
        s1_sc1_l1_helps, s1_sc1_l2_helps, s1_sc1_l3_helps, s1_sc1_l4_helps,
        
        s1_sc2_helps_total,
        s1_sc2_l1_helps, s1_sc2_l2_helps, s1_sc2_l3_helps, s1_sc2_l4_helps,

        s2_sc1_helps_total,
        s2_sc1_l1_helps, s2_sc1_l2_helps, s2_sc1_l3_helps, s2_sc1_l4_helps,

        s2_sc2_helps_total,        
        s2_sc2_l1_helps, s2_sc2_l2_helps, s2_sc2_l3_helps, s2_sc2_l4_helps
    ])

    return vector

def get_score_vector(intera):
    vector = []

    s1_sc1_l1_score = get_performance(intera, 1, 1)
    s1_sc1_l2_score = get_performance(intera, 1, 2)
    s1_sc1_l3_score = get_performance(intera, 1, 3)
    s1_sc1_l4_score = get_performance(intera, 1, 4)
    s1_sc1_score_total = (s1_sc1_l1_score + s1_sc1_l2_score + s1_sc1_l3_score + s1_sc1_l4_score)/4

    s1_sc2_l1_score = get_performance(intera, 2, 1)
    s1_sc2_l2_score = get_performance(intera, 2, 2)
    s1_sc2_l3_score = get_performance(intera, 2, 3)
    s1_sc2_l4_score = get_performance(intera, 2, 4)
    s1_sc2_score_total = (s1_sc2_l1_score + s1_sc2_l2_score + s1_sc2_l3_score + s1_sc2_l4_score)/4

    s2_sc1_l1_score = get_performance(intera, 1, 1, 2)
    s2_sc1_l2_score = get_performance(intera, 1, 2, 2)
    s2_sc1_l3_score = get_performance(intera, 1, 3, 2)
    s2_sc1_l4_score = get_performance(intera, 1, 4, 2)
    s2_sc1_score_total = (s2_sc1_l1_score + s2_sc1_l2_score + s2_sc1_l3_score + s2_sc1_l4_score)/4

    s2_sc2_l1_score = get_performance(intera, 2, 1, 2)
    s2_sc2_l2_score = get_performance(intera, 2, 2, 2)
    s2_sc2_l3_score = get_performance(intera, 2, 3, 2)
    s2_sc2_l4_score = get_performance(intera, 2, 4, 2)
    s2_sc2_score_total = (s2_sc2_l1_score + s2_sc2_l2_score + s2_sc2_l3_score + s2_sc2_l4_score)/4

    
    vector.extend([
        s1_sc1_score_total,
        s1_sc1_l1_score, s1_sc1_l2_score, s1_sc1_l3_score, s1_sc1_l4_score,
        
        s1_sc2_score_total,
        s1_sc2_l1_score, s1_sc2_l2_score, s1_sc2_l3_score, s1_sc2_l4_score,

        s2_sc1_score_total,
        s2_sc1_l1_score, s2_sc1_l2_score, s2_sc1_l3_score, s2_sc1_l4_score,

        s2_sc2_score_total,        
        s2_sc2_l1_score, s2_sc2_l2_score, s2_sc2_l3_score, s2_sc2_l4_score
    ])

    return vector

def get_analysis_vector(user_data):
    survey = user_data["survey"]
    intera = user_data["data"]

    roma            = 0 if survey["Roma"] == "No" else 1 if survey["Roma"] == "Yes" else -1
    adopted         = 0 if survey["Adopted"] == "No" else 1 if survey["Adopted"] == "Yes" else -1
    age             = int(survey["Age"]) if survey["Age"] !="" else -1
    migration_age   = int(survey["MigrationAge"]) if survey["MigrationAge"] != "" else -1
    sex             = 0 if survey["Sex"] == "Boy" else 1
    migrant         = 0 if survey["MigrantBackground"] == "No migratory background" or survey["MigrantBackground"] == "" else 1
    ethnicity       = get_one_hot(survey["Ethnicity"], "etnicity")
    ethnicityMacro  = get_one_hot(survey["EthnicityMacro"], "etnicityMacro")
    migratory       = get_one_hot(survey["MigrantBackground"], "migrant")

    """
    s1_sc1_total_score_total,
    s1_sc1_max_score_total,
    s1_sc1_questions_total,
    s1_sc1_l1_total_score,
    s1_sc1_l2_total_score,
    s1_sc1_l3_total_score,
    s1_sc1_l4_total_score,
    s1_sc1_l1_max_score,
    s1_sc1_l2_max_score,
    s1_sc1_l3_max_score,
    s1_sc1_l4_max_score,
    s1_sc1_l1_questions,
    s1_sc1_l2_questions,
    s1_sc1_l3_questions,
    s1_sc1_l4_questions,
    s1_sc2_total_score_total,
    s1_sc2_max_score_total,
    s1_sc2_questions_total,
    s1_sc2_l1_total_score,
    s1_sc2_l2_total_score,
    s1_sc2_l3_total_score,
    s1_sc2_l4_total_score,
    s1_sc2_l1_max_score,
    s1_sc2_l2_max_score,
    s1_sc2_l3_max_score,
    s1_sc2_l4_max_score,
    s1_sc2_l1_questions,
    s1_sc2_l2_questions,
    s1_sc2_l3_questions,
    s1_sc2_l4_questions,
    """

    demographic_vector = [age,migration_age,sex,roma,adopted,migrant]
    demographic_vector.extend(ethnicity)
    demographic_vector.extend(ethnicityMacro)
    demographic_vector.extend(migratory)


    interaction_vector = get_interaction_vector(intera)
    interaction_vector.extend(get_help_vector(intera))

    score_vector = get_score_vector(intera)

    #Full vector without score
    full_vectorwos = []
    full_vectorwos.extend(demographic_vector)
    full_vectorwos.extend(interaction_vector)

    #Full vector with score
    full_vectorwts = []
    full_vectorwts.extend(demographic_vector)
    full_vectorwts.extend(interaction_vector)
    full_vectorwts.extend(score_vector)

    return demographic_vector, interaction_vector, full_vectorwos, full_vectorwts

def get_analysis_vector_set(data):
    demo_analysis_data , intr_analysis_data, fwos_analysis_data , fwts_analysis_data = [],[],[],[]
    analysis_user = []
    for row in data: 
        demo_vector, intr_vector, full_vectorwos, full_vectorwts = get_analysis_vector(row)
        demo_analysis_data.append(demo_vector)
        intr_analysis_data.append(intr_vector)
        fwos_analysis_data.append(full_vectorwos)
        fwts_analysis_data.append(full_vectorwts)
        analysis_user.append(row["student"])

    return demo_analysis_data, intr_analysis_data, fwos_analysis_data, fwts_analysis_data, analysis_user

def get_analysis_vector_set_session(data,session):
    intr_analysis_data, fwos_analysis_data, fwts_analysis_data = [], [], []
    for row in data: 
        _, _, _, full_vectorwts = get_analysis_vector(row)

        demo = full_vectorwts[:48]             # First 48 values
        intra = full_vectorwts[48:48+80]       # Next 80 values
        score = full_vectorwts[48+80:]         # Last 20 values
        session1 = intra[:30] + intra[60:70]  # first 30 and 10 values for session1
        session2 = intra[30:60] + intra[70:]  # next 30 and last 10 values for session2

        score1 = score[:10] # First 10
        score2 = score[10:] # Last 10

        if session == 1:
            intr_analysis_data.append(session1)
            fwos_analysis_data.append(demo + session1)
            fwts_analysis_data.append(demo + session1 + score1)
        else:
            intr_analysis_data.append(session2)
            fwos_analysis_data.append(demo + session2)
            fwts_analysis_data.append(demo + session2+ score2)

    return intr_analysis_data, fwos_analysis_data, fwts_analysis_data


def get_performance_from_score(value):
    if value>0.66:
        return 1
    return 0

def determine_number_of_clusters(data,type):
    cluster = 0
    silhouettes = []
    maxcluster = 20 if len(data) > 10 else len(data)
    for i in range(3, maxcluster):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data)
        labels = kmeans.labels_
        score = silhouette_score(data, labels)
        silhouettes.append(score)
    cluster = silhouettes.index(max(silhouettes))+3

    wcss = []
    for i in range(3, maxcluster):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    return cluster

def check_PCA(data,type):
    pca = PCA()
    pca.fit(data)
    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(exp_var_cumul >= 0.95) + 1

    return num_components

def clean_non_unique(data):
    data_np = np.array(data)
    # Find columns with more than one unique value
    non_constant_cols = np.any(data_np != data_np[0, :], axis=0)
    # Filter out constant columns
    return data_np[:, non_constant_cols]

def perform_clustering(data,cluster):

    kmeans = KMeans(n_clusters=cluster)
    kmeans.fit(data)
    labels = kmeans.labels_

    return labels

def run_clustering_analysis(data, mode="demographics + interactions + scores"):
    # Get all vectors and analysis users
    demo_data, intr_data, fwos_data, fwts_data, analysis_user = get_analysis_vector_set(data)
    intr_S1, fwos_S1, fwts_S1 = get_analysis_vector_set_session(data, session=1)
    intr_S2, fwos_S2, fwts_S2 = get_analysis_vector_set_session(data, session=2)

    # Mapping of available modes
    mode_map = {
        "demo_vector": demo_data,
        "intr_vector": intr_data,
        "full_vectorwos": fwos_data,
        "full_vectorwts": fwts_data,
        "intr_vector_S1": intr_S1,
        "full_vectorwos_S1": fwos_S1,
        "full_vectorwts_S1": fwts_S1,
        "intr_vector_S2": intr_S2,
        "full_vectorwos_S2": fwos_S2,
        "full_vectorwts_S2": fwts_S2
    }

    if mode not in mode_map:
        raise ValueError(f"Unsupported mode: {mode}")

    # Clean data and perform PCA
    cleaned = clean_non_unique(mode_map[mode])
    components = check_PCA(cleaned, mode)
    pca = PCA(n_components=components)
    data_reduced = pca.fit_transform(cleaned)

    # Determine number of clusters using your original function
    n_clusters = determine_number_of_clusters(data_reduced, mode)

    # Perform clustering
    labels = perform_clustering(data_reduced, n_clusters)
    _, counts = unique(labels, return_counts=True)
    print(f"{mode} Cluster Distribution: {counts}")

    # Save CSV with Name, Cluster, PC1, PC2
    os.makedirs("clustering_outcomes", exist_ok=True)
    df = pd.DataFrame({
        "Name": analysis_user,
        "Cluster": labels,
        "PC1": data_reduced[:, 0],
        "PC2": data_reduced[:, 1]
    })
    df.to_csv(f"clustering_outcomes/{mode}_clusters.csv", index=False)

    # Create interactive scatter plot
    tsne = TSNE(n_components=2, verbose=1)

    tsne_data = array(data_reduced)
    components = tsne.fit_transform(tsne_data)

    x_values = components[:, 0]
    y_values = components[:, 1]

    fig = px.scatter(
        x=x_values, y=y_values,
        color=labels.astype(str),
        hover_name=analysis_user,
        labels={"x": "PC1", "y": "PC2"},
        title=f"{mode} - Clustering (k={n_clusters})"
    )
    fig.update_traces(marker=dict(size=12, opacity=0.7))

    return {
        "fig": fig,
        "labels": labels,
        "data_reduced": data_reduced,
        "recommended_k": n_clusters,
        "analysis_user": analysis_user
    }
