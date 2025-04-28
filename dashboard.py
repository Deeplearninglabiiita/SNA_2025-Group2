
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
import pickle
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Loading and cleaning the dataset
def load_and_clean_data():
    try:
        data = pd.read_csv('CDR-Updated.csv')
        logging.info(f"Loaded dataset with {len(data)} rows and columns: {list(data.columns)}")
    except FileNotFoundError:
        logging.error("CDR-Updated.csv not found")
        raise FileNotFoundError("CDR-Updated.csv not found")
    except Exception as e:
        logging.error(f"Error loading CSV: {str(e)}")
        raise ValueError(f"Error loading CSV: {str(e)}")

    if data.empty:
        logging.error("Loaded dataset is empty")
        raise ValueError("Loaded dataset is empty. Check CSV content.")

    required_cols = ['Phone Number', 'Churn', 'Account Length', 'VMail Message', 'Day Mins', 'Day Calls', 'Day Charge',
                    'Eve Mins', 'Eve Calls', 'Eve Charge', 'Night Mins', 'Night Calls', 'Night Charge',
                    'Intl Mins', 'Intl Calls', 'Intl Charge', 'CustServ Calls']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        logging.error(f"Missing columns: {missing_cols}")
        raise ValueError(f"Missing columns: {missing_cols}")

    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].str.strip()
    logging.info("Stripped whitespace from string columns")

    data['Phone Number'] = data['Phone Number'].fillna('000-0000')
    data['Churn'] = data['Churn'].fillna('FALSE')
    logging.info("Filled missing values in critical columns")

    numeric_cols = ['Account Length', 'VMail Message', 'Day Mins', 'Day Calls', 'Day Charge',
                    'Eve Mins', 'Eve Calls', 'Eve Charge', 'Night Mins', 'Night Calls',
                    'Night Charge', 'Intl Mins', 'Intl Calls', 'Intl Charge', 'CustServ Calls']
    for col in numeric_cols:
        data[col] = data[col].fillna(0)
    logging.info(f"Filled missing values with 0 for numeric columns")

    phone_regexes = [
        r'^\d{3}-\d{4}$',
        r'^\(\d{3}\)\s*\d{3}-\d{4}$',
        r'^\d{3}-\d{3}-\d{4}$',
        r'^\d{10}$',
        r'^\d{3}\.\d{3}\.\d{4}$',
    ]
    valid_rows = pd.Series(False, index=data.index)
    for regex in phone_regexes:
        valid_rows |= data['Phone Number'].str.match(regex, na=False)
    
    if valid_rows.sum() > 0:
        data = data[valid_rows]
        logging.info(f"After phone number filter, {len(data)} rows remain")
    else:
        logging.warning("Phone number filter would remove all rows. Skipping phone number validation.")

    if data.empty:
        logging.error("No rows remain after phone number filtering")
        raise ValueError("No rows remain after phone number filtering")

    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
    logging.info(f"Converted numeric columns to numeric type")

    data['Churn'] = data['Churn'].astype(str).str.upper().map({'TRUE': True, 'FALSE': False, '1': True, '0': False})
    data['Churn'] = data['Churn'].fillna(False)

    if data.empty:
        logging.error("No rows remain after final cleaning")
        raise ValueError("No rows remain after final cleaning")

    logging.info(f"Final cleaned dataset has {len(data)} rows")
    return data

# Feature engineering
def engineer_features(data):
    data['Total Calls'] = data['Day Calls'] + data['Eve Calls'] + data['Night Calls'] + data['Intl Calls']
    data['Total Mins'] = data['Day Mins'] + data['Eve Mins'] + data['Night Mins'] + data['Intl Mins']
    data['Avg Call Duration'] = data['Total Mins'] / data['Total Calls'].replace(0, 1)
    data['Avg Intl Call Duration'] = data['Intl Mins'] / data['Intl Calls'].replace(0, np.nan)
    data['Avg Intl Call Duration'] = data['Avg Intl Call Duration'].fillna(0)
    logging.info(f"Feature engineering complete, {len(data)} rows remain")
    return data

# Social network features
def social_network_features(data, max_nodes=50, edge_prob=0.1, random_state=42):
    np.random.seed(random_state)
    G = nx.Graph()
    
    # Sample phone numbers
    sampled_phones = data['Phone Number'].sample(n=min(max_nodes, len(data)), random_state=random_state).tolist()
    
    # Add nodes to the graph with behavior attributes
    for phone in sampled_phones:
        behavior = data.loc[data['Phone Number'] == phone, 'Behavior'].iloc[0] if 'Behavior' in data.columns else 'Unknown'
        G.add_node(phone, behavior=behavior)
    
    # Randomly add edges with probability edge_prob
    for i in range(len(sampled_phones)):
        for j in range(i + 1, len(sampled_phones)):
            if np.random.rand() < edge_prob:
                G.add_edge(sampled_phones[i], sampled_phones[j], weight=1)
    
    # Compute features
    degree_centrality = nx.degree_centrality(G)
    clustering_coeff = nx.clustering(G)
    
    # Map back to data
    data['Degree Centrality'] = data['Phone Number'].map(degree_centrality).fillna(0)
    data['Clustering Coefficient'] = data['Phone Number'].map(clustering_coeff).fillna(0)
    
    logging.info(f"Social network features added with random graph (p={edge_prob}), {len(data)} rows remain")
    return data, G

# Assign behavior classes
def assign_classes(data):
    def classify_behavior(row):
        if row['Total Calls'] > 15 and row['Avg Call Duration'] < 1:
            return 'Spam'
        elif row['Intl Calls'] > 5 and row['Avg Intl Call Duration'] < 2 and row['CustServ Calls'] > 3:
            return 'Scam'
        elif row['Total Calls'] > 10 and row['Avg Call Duration'] < 3 and row['Night Calls'] < 2:
            return 'Business'
        elif row['Total Calls'] < 5 and row['Avg Call Duration'] > 10 and row['Night Mins'] > row['Day Mins']:
            return 'Family'
        elif 3 <= (row['Total Calls'] / 30) <= 10 and 3 <= row['Avg Call Duration'] <= 10:
            return 'Regular'
        else:
            return 'Other'
    
    data['Behavior'] = data.apply(classify_behavior, axis=1)
    logging.info(f"Behavior classes assigned, {len(data)} rows remain")
    return data

# Train model
def train_model(data):
    features = ['Day Mins', 'Eve Mins', 'Night Mins', 'Intl Mins',
                'Day Calls', 'Eve Calls', 'Night Calls', 'Intl Calls',
                'CustServ Calls', 'Total Calls', 'Total Mins',
                'Avg Call Duration', 'Avg Intl Call Duration',
                'Degree Centrality', 'Clustering Coefficient']
    
    X = data[features]
    y = data['Behavior']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    
    # Train model
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    })
    feature_importance.to_csv('feature_importance.csv', index=False)
    
    # Save model, label encoder, and scaler
    with open('hoax_call_detector.pkl', 'wb') as f:
        pickle.dump({'model': model, 'label_encoder': le, 'scaler': scaler}, f)
    
    logging.info("Model training complete")
    return model, le, scaler, feature_importance

# Create interactive 3D network plot
def plot_network(G, phone_numbers):
    pos = nx.spring_layout(G, dim=3, seed=42)
    
    # Edges
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Nodes
    node_x = []
    node_y = []
    node_z = []
    node_text = []
    node_ids = []
    behavior_map = {'Spam': 0, 'Scam': 1, 'Business': 2, 'Family': 3, 'Regular': 4, 'Other': 5, 'Unknown': -1}
    node_colors = []
    
    for node in G.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        behavior = G.nodes[node].get('behavior', 'Unknown')
        node_text.append(f"Phone: {node}<br>Behavior: {behavior}<br>Degree: {G.degree[node]}")
        node_colors.append(behavior_map[behavior])
        node_ids.append(node)
    
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        customdata=[{'phone': node, 'behavior': G.nodes[node].get('behavior', 'Unknown'), 'degree': G.degree[node]} for node in G.nodes()],
        marker=dict(
            showscale=True,
            colorscale='RdYlGn',
            reversescale=True,
            size=8,
            color=node_colors,
            colorbar=dict(
                thickness=15,
                title='Behavior',
                xanchor='left',
                tickvals=list(behavior_map.values()),
                ticktext=list(behavior_map.keys())
            ),
            line=dict(width=2, color='DarkSlateGrey'),
            opacity=0.8
        ),
        hoverlabel=dict(bgcolor='black', font_size=12)
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(
                            text='3D Simulated Call Network (Colored by Behavior)',
                            font=dict(size=16)
                        ),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        scene=dict(
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title='')
                        ),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    ))
    
    fig.update_traces(
        selector=dict(type='scatter3d', mode='markers'),
        hovertemplate='%{text}',
        marker=dict(
            sizemode='diameter',
            sizeref=0.1,
            sizemin=4
        )
    )
    
    return fig

# Streamlit dashboard
def main():
    st.set_page_config(page_title="Hoax Call Detection Dashboard", layout="wide")
    st.title("📞 Hoax Call Detection Dashboard")
    
    # Custom CSS to increase sidebar font size
    st.markdown("""
        <style>
        .sidebar .sidebar-content {
            font-size: 18px; /* Increase font size for sidebar options */
        }
        .sidebar .stRadio > label {
            font-size: 18px; /* Specifically target radio button labels */
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state for node selection
    if 'selected_node' not in st.session_state:
        st.session_state.selected_node = None
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dataset Overview", "Data Input & Prediction", "Model Visualizations", "Workflow"])
    
    # Load or train model
    data = None
    G = None
    model = None
    le = None
    scaler = None
    if os.path.exists('hoax_call_detector.pkl'):
        with open('hoax_call_detector.pkl', 'rb') as f:
            saved_data = pickle.load(f)
        model = saved_data['model']
        le = saved_data['label_encoder']
        scaler = saved_data['scaler']
        data = load_and_clean_data()
        data = engineer_features(data)
        data = assign_classes(data)
        data, G = social_network_features(data)
    else:
        data = load_and_clean_data()
        data = engineer_features(data)
        data = assign_classes(data)
        data, G = social_network_features(data)
        model, le, scaler, feature_importance = train_model(data)
    
    if page == "Dataset Overview":
        st.header("Dataset Overview")
        st.write("""
            The dataset contains telecom call records used to detect hoax calls. It includes information about phone numbers, call durations, call counts, charges, and customer service interactions. The model predicts behaviors such as 'Spam', 'Scam', 'Business', 'Family', 'Regular', or 'Other' based on these features. Below is a sample of the dataset (first 5 rows):
        """)
        st.dataframe(data.head(), use_container_width=True)
        st.write(f"**Total Rows**: {len(data)}")
        st.write(f"**Columns**: {', '.join(data.columns)}")
    
    elif page == "Data Input & Prediction":
        st.header("Customer Data Input & Prediction")
        
        with st.form("customer_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                phone_number = st.text_input("Phone Number", "123-4567")
                churn = st.selectbox("Churn", [False, True])
                account_length = st.number_input("Account Length", min_value=0, value=100)
                vmail_message = st.number_input("Voicemail Messages", min_value=0, value=0)
                day_mins = st.number_input("Day Minutes", min_value=0.0, value=180.0)
                
            with col2:
                day_calls = st.number_input("Day Calls", min_value=0, value=100)
                day_charge = st.number_input("Day Charge", min_value=0.0, value=30.0)
                eve_mins = st.number_input("Evening Minutes", min_value=0.0, value=200.0)
                eve_calls = st.number_input("Evening Calls", min_value=0, value=100)
                eve_charge = st.number_input("Evening Charge", min_value=0.0, value=17.0)
                
            with col3:
                night_mins = st.number_input("Night Minutes", min_value=0.0, value=200.0)
                night_calls = st.number_input("Night Calls", min_value=0, value=100)
                night_charge = st.number_input("Night Charge", min_value=0.0, value=9.0)
                intl_mins = st.number_input("International Minutes", min_value=0.0, value=10.0)
                intl_calls = st.number_input("International Calls", min_value=0, value=3)
                intl_charge = st.number_input("International Charge", min_value=0.0, value=2.7)
                custserv_calls = st.number_input("Customer Service Calls", min_value=0, value=1)
            
            submitted = st.form_submit_button("Predict Behavior")
        
        if submitted:
            input_data = pd.DataFrame({
                'Phone Number': [phone_number],
                'Churn': [churn],
                'Account Length': [account_length],
                'VMail Message': [vmail_message],
                'Day Mins': [day_mins],
                'Day Calls': [day_calls],
                'Day Charge': [day_charge],
                'Eve Mins': [eve_mins],
                'Eve Calls': [eve_calls],
                'Eve Charge': [eve_charge],
                'Night Mins': [night_mins],
                'Night Calls': [night_calls],
                'Night Charge': [night_charge],
                'Intl Mins': [intl_mins],
                'Intl Calls': [intl_calls],
                'Intl Charge': [intl_charge],
                'CustServ Calls': [custserv_calls]
            })
            
            input_data = engineer_features(input_data)
            input_data, _ = social_network_features(input_data)
            
            features = ['Day Mins', 'Eve Mins', 'Night Mins', 'Intl Mins',
                        'Day Calls', 'Eve Calls', 'Night Calls', 'Intl Calls',
                        'CustServ Calls', 'Total Calls', 'Total Mins',
                        'Avg Call Duration', 'Avg Intl Call Duration',
                        'Degree Centrality', 'Clustering Coefficient']
            
            X_input = input_data[features]
            X_input_scaled = scaler.transform(X_input)
            prediction_encoded = model.predict(X_input_scaled)[0]
            prediction = le.inverse_transform([prediction_encoded])[0]
            
            # Display prediction with warning for Spam/Scam
            if prediction in ['Spam', 'Scam']:
                st.warning(f"⚠️ Suspicious Hoax Call Detected! Predicted Behavior: **{prediction}**")
            else:
                st.success(f"✅ Not a Hoax Call. Predicted Behavior: **{prediction}**")
    
    elif page == "Model Visualizations":
        st.header("Model Performance Visualizations")
        
        # Feature Importance
        if os.path.exists('feature_importance.csv'):
            feature_importance = pd.read_csv('feature_importance.csv')
            st.subheader("Feature Importance")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feature_importance.sort_values('Importance', ascending=False), 
                       x='Importance', y='Feature', ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
        
        # Confusion Matrix Image
        st.subheader("Confusion Matrix")
        if os.path.exists('cf_matrix.jpg'):
            st.image('cf_matrix.jpg', caption="Confusion Matrix for Model Predictions", use_container_width=True)
        else:
            st.error("Confusion matrix image (cf_matrix.jpg) not found in the working directory.")

         # Classification Report
        st.subheader("Classification Report")
        if os.path.exists('csr.jpg'):
            st.image('csr.jpg', caption="Classification Report for Model (XGBoost) Predictions", use_container_width=True)
        else:
            st.error("Classification Report (csr.jpg) not found in the working directory.")
        # Network Visualization
        if G is not None:
            st.subheader("Interactive 3D Call Network")
            st.write("This 3D plot shows the simulated call network. Nodes represent phone numbers, colored by behavior (e.g., Spam, Scam, Business, Family, Regular, Other), floating in 3D space. Edges indicate simulated connections. Hover over nodes to enlarge them and see details; click a node to display its properties below. Rotate and zoom to explore.")
            
            fig = plot_network(G, list(G.nodes()))
            st.plotly_chart(fig, use_container_width=True, key="network_plot")
            
            st.subheader("Selected Node Properties")
            if st.session_state.selected_node:
                node_info = st.session_state.selected_node
                st.write(f"**Phone Number**: {node_info['phone']}")
                st.write(f"**Behavior**: {node_info['behavior']}")
                st.write(f"**Degree**: {node_info['degree']}")
            else:
                st.write("Click a node in the network to display its properties.")
    
    elif page == "Workflow":
        st.header("Workflow")
        st.write("The following flowchart illustrates the workflow of the hoax call detection process, from data loading to model prediction and visualization.")
        if os.path.exists('beautiful_hoax_call_analysis_flowchart (2).png'):
            st.image('beautiful_hoax_call_analysis_flowchart (2).png', caption="Hoax Call Detection Workflow", use_container_width=True)
        else:
            st.error("Workflow image (beautiful_hoax_call_analysis_flowchart (2).png) not found in the working directory.")

if __name__ == "__main__":
    main()