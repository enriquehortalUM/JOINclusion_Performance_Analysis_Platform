import streamlit as st
import os
from numpy import array
import pandas as pd
from sklearn.manifold import TSNE
import json
import joblib
from clustering import run_clustering_analysis, perform_clustering
from data_analysis_teachers import get_feature_vectors, filter_non_played
from performance_prediction import predict_student_performance_all_features
import plotly.express as px
import plotly.graph_objects as go


# --- Dashboard 3: Teachers Dashboard ---
def teachers_dashboard():
    st.title("Teachers Dashboard")

    # Upload JSON data file
    json_file = st.file_uploader("Upload student data (JSON)", type="json")

    DEFAULT_JSON_PATH = "data/sample.json"
    if json_file is not None:
        data = json.load(json_file)
        st.success("Custom student data loaded.")
    elif os.path.exists(DEFAULT_JSON_PATH):
        with open(DEFAULT_JSON_PATH, "r") as f:
            data = json.load(f)
        st.info("Using default student data.")
    else:
        data = None
        st.error("No student data available.")

    if data:
        size = len(data)
        index=0
        data = filter_non_played(data)[index:index+size]

        interaction_vectors, full_vectors, usernames, feature_names = get_feature_vectors(data)

        df = pd.DataFrame(full_vectors, columns=feature_names)

        best_students = df.nlargest(10, "best_score")[["name", "best_score"]]
        worst_students = df.nsmallest(10, "best_score")[["name", "best_score"]]
        col1, col2 = st.columns(2)

        scenarios = ["Scenario 1", "Scenario 2"]
        for scenario in scenarios:
            score_column = "best_score_s1" if scenario == "Scenario 1" else "best_score_s2"
            scenario_df = df[["name", score_column]].sort_values(score_column, ascending=False)
            fig = px.bar(
                scenario_df,
                x="name",
                y=score_column,
                color=score_column,
                title=f"Scores for {scenario}"
            )
            fig.update_layout(xaxis_title=None)
            st.plotly_chart(fig)


        with col1:
            st.write("Highest score students:")
            st.write(best_students)

        with col2:
            st.write("Lowest score students:")
            st.write(worst_students)

        if 'student_names' not in st.session_state:
            st.session_state.student_names = df['name'].unique()

        selected_student = st.selectbox(
            'Select a student:',
            df['name'].unique(),
            index=0
        )

        student_data = df[df['name'] == selected_student]

        features_to_display = ['best_score', 'total_time_played', 'total_interactions', "total_helps"]

        cols = [st.columns(2), st.columns(2)]  

        for idx, feature in enumerate(features_to_display):
            row = idx // 2  
            col = idx % 2  
            with cols[row][col]:
                student_value = student_data[feature].iloc[0]
                fig = go.Figure()

                nbins = min(len(df[feature].unique()), 10)
                fig.add_trace(go.Histogram(
                    x=df[feature],
                    name='Distribution',
                    marker=dict(color='lightblue'),
                    opacity=0.75,
                    nbinsx=nbins
                ))
                fig.add_trace(go.Scatter(
                    x=[student_value],
                    y=[0],
                    mode='markers',
                    marker=dict(size=32, color='red', symbol='line-ns-open', line=dict(width=2)),
                    name=selected_student,
                    hovertemplate=f"{selected_student}: <b>{student_value:,.0f}</b><extra></extra>"
                ))
                fig.update_layout(
                    title=dict(
                        text=f"{feature.replace('_', ' ').title()}",
                        x=0.25
                    ),
                    xaxis=dict(title=feature.replace('_', ' ').title()),
                    yaxis=dict(title='Count'),
                    height=300,
                    width=300,
                    margin=dict(l=40, r=40, t=40, b=40),
                    showlegend=False,
                    bargap=0.1
                )
                st.plotly_chart(fig, use_container_width=False)

        features = ["best_score","best_score_s1","best_score_s2","total_time_played","total_interactions","total_helps"]
        optional = ["total_help_s1","total_help_s2","total_character_interactions_s1","total_character_interactions_s2","total_change_scene_interactions_s1","total_change_scene_interactions_s2","total_movement_interactions_s1","total_movement_interactions_s2"]

        include_optional = st.pills(
            '',
            options=['Overall','Detailed'],
            default = 'Overall'
        )
        if include_optional == 'Detailed':
            features.extend(optional)

        percentiles = {}
        for feature in features:
            student_value = student_data[feature].iloc[0]
            percentile = (df[feature] <= student_value).mean() * 100
            percentiles[feature] = percentile

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[percentiles[feat] for feat in features],
            theta=features,
            fill='toself',
            name=selected_student,
            hovertemplate=f"{selected_student}"
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    tickfont=dict(size=12, color='gray'),
                    range=[0, 100]
                )
            ),
            showlegend=False,
            title=f"Percentile Rankings for {selected_student}",
            height=500,
            width=800
        )
        st.plotly_chart(fig_radar, use_container_width=False)
 
# --- Dashboard: Clustering Dashboard ---
def clustering_dashboard():
    st.title("Clustering Analysis Dashboard")

    # Upload JSON data file
    json_file = st.file_uploader("Upload student data (JSON)", type="json")

    DEFAULT_JSON_PATH = "data/sample.json"
    if json_file is not None:
        json_data = json.load(json_file)
        st.success("Custom student data loaded.")
    elif os.path.exists(DEFAULT_JSON_PATH):
        with open(DEFAULT_JSON_PATH, "r") as f:
            json_data = json.load(f)
        st.info("Using default student data.")
    else:
        json_data = None
        st.error("No student data available.")

    # Mapping of user-friendly labels to mode keys
    mode_options = {
        "Demographics": "demo_vector",
        "Interactions": "intr_vector",
        "Demographics + Interactions": "full_vectorwos",
        "Demographics + Interactions + Scores": "full_vectorwts",
        "Session 1: Interactions": "intr_vector_S1",
        "Session 1: Demographics + Interactions": "full_vectorwos_S1",
        "Session 1: Demographics + Interactions + Scores": "full_vectorwts_S1",
        "Session 2: Interactions": "intr_vector_S2",
        "Session 2: Demographics + Interactions": "full_vectorwos_S2",
        "Session 2: Demographics + Interactions + Scores": "full_vectorwts_S2"
    }

    selected_label = st.selectbox("Select the analysis type:", list(mode_options.keys()))
    analysis_mode = mode_options[selected_label]

    if json_data:
        try:
            st.success("File loaded!")

            st.write(f"Running analysis with mode: **{selected_label}**")
            results = run_clustering_analysis(json_data, mode=analysis_mode)

            recommended_k = results["recommended_k"]
            data_reduced = results["data_reduced"]
            analysis_user = results["analysis_user"]

            k = st.slider("Select number of clusters (k):", min_value=2, max_value=10, value=recommended_k)

            labels = perform_clustering(data_reduced, k)

            # Create interactive scatter plot
            tsne = TSNE(n_components=2, verbose=1)

            tsne_data = array(data_reduced)
            components = tsne.fit_transform(tsne_data)

            x_values = components[:, 0]
            y_values = components[:, 1]

            fig = px.scatter(
                #x=data_reduced[:, 0], y=data_reduced[:, 1],
                x=x_values, y=y_values,
                color=labels.astype(str),
                hover_name=analysis_user,
                labels={"x": "t-SNE 1", "y": "t-SNE 2"},
                title=f"{selected_label} - Clustering (k={k})"
            )
            fig.update_traces(marker=dict(size=12, opacity=0.7))

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")

# --- Dashboard: Prediction Dashboard ---
def prediction_dashboard():
    st.title("Collaborative Performance Predictor")

    # Paths to default files
    DEFAULT_MODEL_PATH = "models/model_XGBoost.joblib"
    DEFAULT_JSON_PATH = "data/sample.json"

    # Upload model file
    model_file = st.file_uploader("Upload trained ML model (.joblib)", type="joblib")

    if model_file is not None:
        model = joblib.load(model_file)
        st.success("Custom model loaded.")
    elif os.path.exists(DEFAULT_MODEL_PATH):
        model = joblib.load(DEFAULT_MODEL_PATH)
        st.info("Using default model.")
    else:
        model = None
        st.error("No model available.")

    # Upload JSON data file
    json_file = st.file_uploader("Upload student data (JSON)", type="json")

    if json_file is not None:
        raw_data = json.load(json_file)
        st.success("Custom student data loaded.")
    elif os.path.exists(DEFAULT_JSON_PATH):
        with open(DEFAULT_JSON_PATH, "r") as f:
            raw_data = json.load(f)
        st.info("Using default student data.")
    else:
        raw_data = None
        st.error("No student data available.")

    # Proceed if both model and data are available
    if model and raw_data:
        # Extract student names
        student_names = [student["student"] for student in raw_data]

        # Student selection
        selected_student = st.selectbox("Select a student to predict", student_names)

        # Teammate selection (excluding selected student)
        possible_teammates = [name for name in student_names if name != selected_student]
        selected_teammates = st.multiselect(
            "Select up to 3 teammates",
            possible_teammates,
            max_selections=3
        )

        # Trigger prediction only when 3 teammates are selected
        if len(selected_teammates) == 3:
            try:
                prediction = predict_student_performance_all_features(
                    model, raw_data, selected_student, teammates=selected_teammates
                )
                label = "High performance" if prediction == 1 else "Low performance"
                st.subheader(f"Prediction for {selected_student}:")
                st.markdown(f"**Expected collaborative performance:** {label}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.warning("Please select exactly 3 teammates to run the prediction.")


# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Students Dashboard", "Prediction", "Clustering"))

# --- Routing ---
if page == "Students Dashboard":
    teachers_dashboard()
elif page == "Prediction":
    prediction_dashboard()
elif page == "Clustering":
    clustering_dashboard()