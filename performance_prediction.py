import numpy as np

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

def get_performance_from_score(value):
    if value>0.66:
        return 1
    return 0

def get_collab_compatible_vector(data):
    interaction_vector = get_interaction_vector(data)
    help_vector = get_help_vector(data)
    score_vector = get_score_vector(data)
    s1_sc1_total = interaction_vector[0] + interaction_vector[1] + interaction_vector[2]
    s1_sc2_total = interaction_vector[15] + interaction_vector[16] + interaction_vector[17]
    s2_sc1_total = interaction_vector[30] + interaction_vector[31] + interaction_vector[32]
    s2_sc2_total = interaction_vector[45] + interaction_vector[46] + interaction_vector[47]
    s1_sc1_total_helps = help_vector[0]
    s1_sc2_total_helps = help_vector[5]
    s2_sc1_total_helps = help_vector[10]
    s2_sc2_total_helps = help_vector[15]
    s1_sc1_total_score = score_vector[0]
    s1_sc2_total_score = score_vector[5]
    s2_sc1_total_score = score_vector[10]
    s2_sc2_total_score = score_vector[15]

    game1 = [s1_sc1_total+s1_sc2_total, s1_sc1_total_helps+s1_sc2_total_helps, get_performance_from_score((s1_sc1_total_score+s1_sc2_total_score)/2)]
    game2 = [s2_sc1_total+s2_sc2_total, s2_sc1_total_helps+s2_sc2_total_helps, get_performance_from_score((s2_sc1_total_score+s2_sc2_total_score)/2)]
    return game1, game2

def get_collab_performance(data,session = 1):
    value = 0
    if session == 1:
        if len(data) > 0:
            scr = data[0]["scores"]["breakdown"][2]["total_score"]
            max = data[0]["scores"]["breakdown"][2]["max_score"]
            que = data[0]["scores"]["breakdown"][2]["total_question"] 
        scr, max, que = 0, 0, 0      
    else: 
        count = 0
        scr, max, que = 0, 0, 0
        for row in data:
            if count > 0:
                scr += row["scores"]["breakdown"][2]["max_score"]
                max += row["scores"]["breakdown"][2]["max_score"]
                que += row["scores"]["breakdown"][2]["total_question"]    
            count += 1
    value = scr/max if que > 0 else 0
    return value

def get_data_from_name(data,name):
    for row in data:
        if row["student"] == name:
            return row
    return {"data":[]}


def predict_student_performance_all_features(model, raw_data, selected_student, teammates):
    '''
    Predict collaborative performance using all features (full vector).
    Expects a trained model and a student record as parsed from JSON.
    '''
    vector = get_custom_collaboration_set(raw_data, selected_student, teammates)
    vector = np.array(vector[0])
    feature_names = [
        'S1_Interaction', 'S1_HelpRequests', 'S1_Score',
        'S2_Interaction', 'S2_HelpRequests', 'S2_Score',
        'S3_Interaction', 'S3_HelpRequests', 'S3_Score',
        'S4_Interaction', 'S4_HelpRequests', 'S4_Score'
    ]

    # Prepare 2D input array for model prediction
    X = np.array([vector], dtype=np.float32)
    prediction = model.predict(X)[0]
    return prediction

def get_custom_collaboration_set(data, student_name, teammates_s1=[]):
    collab_data = []
    
    # Get student record
    student_data = next((row for row in data if row["student"] == student_name), None)

    # Find the main student
    student_record = get_data_from_name(data, student_name)
    if not student_record or "data" not in student_record:
        return []

    # Session 1
    if len(teammates_s1) > 0:
        collab_row = []
        game1, _ = get_collab_compatible_vector(student_record["data"])
        score1 = get_performance_from_score(get_collab_performance(student_record["data"]))
        collab_row.extend(game1)

        for member in teammates_s1[:3]:  # Use only up to 3 teammates
            teammate_record = get_data_from_name(data, member)
            if teammate_record and "data" in teammate_record:
                game1t, _ = get_collab_compatible_vector(teammate_record["data"])
                collab_row.extend(game1t)
        collab_data.append(collab_row)

    return collab_data
