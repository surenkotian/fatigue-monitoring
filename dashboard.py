import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time
import pickle
import os
import hashlib
import yaml
from io import BytesIO
import base64
import joblib
import uuid

# -------------------- AUTHENTICATION --------------------

def load_config():
    """Load the configuration file containing user credentials"""
    try:
        with open("config.yaml") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        # Create default config if it doesn't exist
        default_config = {
            "credentials": {
                "admin": {
                    "username": "admin",
                    "password": hashlib.sha256("admin123".encode()).hexdigest(),
                    "role": "admin"
                }
            }
        }
        with open("config.yaml", "w") as file:
            yaml.dump(default_config, file)
        return default_config

def authenticate(username, password):
    """Authenticate user against stored credentials"""
    config = load_config()
    if username in config["credentials"]:
        stored_password = config["credentials"][username]["password"]
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        if stored_password == hashed_password:
            return True, config["credentials"][username]["role"]
    return False, None

def register_user(username, password, role="user"):
    """Register a new user"""
    config = load_config()
    if username in config["credentials"]:
        return False, "Username already exists"
    
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    config["credentials"][username] = {
        "username": username,
        "password": hashed_password,
        "role": role
    }
    
    with open("config.yaml", "w") as file:
        yaml.dump(config, file)
    return True, "User registered successfully"

# -------------------- DATA FUNCTIONS --------------------

@st.cache_data(ttl=300)  # Cache data for 5 minutes
def load_data():
    """Load and preprocess the data"""
    try:
        df = pd.read_csv('/app/wfh_fatigue_data.csv')
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Hour'] = df['Timestamp'].dt.hour
        df['Date'] = df['Timestamp'].dt.date
        
        # Calculate additional metrics
        df['Blink_Rate'] = 1 / (df['EAR'] + 0.1)  # Inverse of EAR (higher when eyes more closed)
        df['Yawn_Intensity'] = df['MAR'] * 10  # Scale MAR for better visualization
        
        # Smoothing
        df['EAR_Smoothed'] = df['EAR'].rolling(window=20, min_periods=1).mean()
        df['MAR_Smoothed'] = df['MAR'].rolling(window=20, min_periods=1).mean()
        
        # Efficiency score calculation
        df['EfficiencyScore'] = (df['EAR'] * 0.7) - (df['MAR'] * 0.3)
        
        # Activity level based on eye and mouth movement
        df['Activity'] = df['EAR'].diff().abs() + df['MAR'].diff().abs()
        df['Activity'] = df['Activity'].fillna(0).rolling(window=50, min_periods=1).mean()
        
        # Define fatigue level
        def get_fatigue_level(ear, mar):
            if ear < 0.2:
                return "High Fatigue"
            elif ear < 0.25 or mar > 0.5:
                return "Moderate Fatigue"
            else:
                return "Low Fatigue"
        
        df['FatigueLevel'] = df.apply(lambda row: get_fatigue_level(row['EAR'], row['MAR']), axis=1)
        
        # Define efficiency level
        def efficiency_level(score):
            if score > 0.5:
                return "High"
            elif score > 0.3:
                return "Medium"
            else:
                return "Low"
                
        df['EfficiencyLevel'] = df['EfficiencyScore'].apply(efficiency_level)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Generate dummy data for demonstration if file not found
        return generate_dummy_data()

def generate_dummy_data():
    """Generate dummy data for demonstration"""
    start_date = datetime.now() - timedelta(days=7)
    timestamps = [start_date + timedelta(minutes=i*5) for i in range(2000)]
    
    # Simulate daily patterns (more tired in afternoon)
    ears = []
    mars = []
    states = []
    
    for ts in timestamps:
        hour = ts.hour
        # Lower EAR (more tired) in afternoon
        baseline_ear = 0.3 - 0.05 * (1 if 13 <= hour <= 16 else 0)
        ear = max(0.1, np.random.normal(baseline_ear, 0.05))
        
        # Higher MAR (more yawning) when tired
        baseline_mar = 0.2 + 0.1 * (1 if 13 <= hour <= 16 else 0)
        mar = max(0.1, np.random.normal(baseline_mar, 0.07))
        
        if ear < 0.2:
            state = "Drowsy"
        elif mar > 0.4:
            state = "Yawning"
        else:
            state = "Alert"
            
        ears.append(ear)
        mars.append(mar)
        states.append(state)
    
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "EAR": ears,
        "MAR": mars,
        "State": states
    })
    
    # Add additional fields similar to load_data function
    df['Hour'] = df['Timestamp'].dt.hour
    df['Date'] = df['Timestamp'].dt.date
    df['Blink_Rate'] = 1 / (df['EAR'] + 0.1)
    df['Yawn_Intensity'] = df['MAR'] * 10
    df['EAR_Smoothed'] = df['EAR'].rolling(window=20, min_periods=1).mean()
    df['MAR_Smoothed'] = df['MAR'].rolling(window=20, min_periods=1).mean()
    df['EfficiencyScore'] = (df['EAR'] * 0.7) - (df['MAR'] * 0.3)
    df['Activity'] = df['EAR'].diff().abs() + df['MAR'].diff().abs()
    df['Activity'] = df['Activity'].fillna(0).rolling(window=50, min_periods=1).mean()
    
    # Define fatigue level
    def get_fatigue_level(ear, mar):
        if ear < 0.2:
            return "High Fatigue"
        elif ear < 0.25 or mar > 0.5:
            return "Moderate Fatigue"
        else:
            return "Low Fatigue"
    
    df['FatigueLevel'] = df.apply(lambda row: get_fatigue_level(row['EAR'], row['MAR']), axis=1)
    
    # Define efficiency level
    def efficiency_level(score):
        if score > 0.5:
            return "High"
        elif score > 0.3:
            return "Medium"
        else:
            return "Low"
            
    df['EfficiencyLevel'] = df['EfficiencyScore'].apply(efficiency_level)
    
    return df

def get_recommendations(fatigue_data):
    """Generate personalized recommendations based on fatigue patterns"""
    recommendations = []

    # Calculate metrics for recommendation logic
    drowsy_count = len(fatigue_data[fatigue_data['State'] == 'Drowsy'])
    if len(fatigue_data) == 0:
        drowsy_percentage = 0
    else:
        drowsy_percentage = (drowsy_count / len(fatigue_data)) * 100

    # Time analysis
    if 'Hour' in fatigue_data.columns:
        hour_fatigue = fatigue_data.groupby('Hour')['EAR'].mean().idxmin()

        # Add time-based recommendations
        if 13 <= hour_fatigue <= 15:
            recommendations.append(
                f"🕒 Your lowest alertness occurs around {hour_fatigue}:00. Consider scheduling a short break or power nap after lunch."
            )

    # Fatigue severity
    if drowsy_percentage > 30:
        recommendations.append("⚠️ High fatigue levels detected. Consider longer breaks and ensure you're getting enough sleep.")
    elif drowsy_percentage > 15:
        recommendations.append("⚠️ Moderate fatigue detected. Try the 20-20-20 rule: every 20 minutes, look at something 20 feet away for 20 seconds.")

    # Add general recommendations
    recommendations.append("💧 Stay hydrated! Dehydration can contribute to eye fatigue.")
    recommendations.append("🔆 Adjust your screen brightness to match your environment.")

    return recommendations

def get_chart_theme():
    """Return chart theme consistent with dashboard style"""
    return {
        'bgcolor': 'rgba(17, 17, 17, 0.95)',
        'font_color': '#f0f2f6',
        'grid_color': 'rgba(255, 255, 255, 0.1)',
        'colorscale': 'Viridis'
    }

# -------------------- UI FUNCTIONS --------------------

def get_time_filter_options():
    """Return time filter options for dashboard"""
    return ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time", "Custom Range"]

def apply_time_filter(df, filter_option, start_date=None, end_date=None):
    """Apply selected time filter to dataframe"""
    now = datetime.now()
    
    if filter_option == "Last 24 Hours":
        return df[df['Timestamp'] > (now - timedelta(days=1))]
    elif filter_option == "Last 7 Days":
        return df[df['Timestamp'] > (now - timedelta(days=7))]
    elif filter_option == "Last 30 Days":
        return df[df['Timestamp'] > (now - timedelta(days=30))]
    elif filter_option == "Custom Range" and start_date and end_date:
        end_date = datetime.combine(end_date, datetime.max.time())  # Set to end of day
        return df[(df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)]
    else:
        return df  # All time

def create_download_link(df, filename):
    """Create a download link for the dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    return href

def plot_hourly_heatmap(df):
    """Plot heatmap of fatigue by hour of day"""
    # Prepare data
    hourly_data = df.groupby(['Date', 'Hour'])['EAR'].mean().reset_index()
    pivot_data = hourly_data.pivot(index='Date', columns='Hour', values='EAR')
    
    # Create heatmap figure
    theme = get_chart_theme()
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=[str(date) for date in pivot_data.index],
        colorscale='RdYlGn_r',  # Red for low EAR (tired), green for high EAR (alert)
        reversescale=True,
        showscale=True,
        colorbar=dict(title="Alertness (EAR)"),
    ))
    
    fig.update_layout(
        title="Alertness Heatmap by Hour and Day",
        xaxis_title="Hour of Day",
        yaxis_title="Date",
        font=dict(color=theme['font_color']),
        plot_bgcolor=theme['bgcolor'],
        paper_bgcolor=theme['bgcolor'],
        height=450,
    )
    
    return fig

def plot_session_metrics(df):
    """Plot session metrics over time"""
    # Group by date to get session stats
    daily_metrics = df.groupby('Date').agg({
        'EAR': 'mean',
        'MAR': 'mean',
        'State': lambda x: (x == 'Drowsy').mean() * 100,  # Percentage of drowsy states
        'Timestamp': 'count'  # Number of observations (proxy for session length)
    }).reset_index()
    
    daily_metrics.columns = ['Date', 'Avg EAR', 'Avg MAR', 'Drowsy %', 'Session Length']
    
    # Create subplots with 2 y-axes
    theme = get_chart_theme()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    fig.add_trace(
        go.Scatter(x=daily_metrics['Date'], y=daily_metrics['Avg EAR'], 
                   mode='lines+markers', name='Average EAR',
                   line=dict(color='#3498db', width=2)),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=daily_metrics['Date'], y=daily_metrics['Drowsy %'], 
                   mode='lines+markers', name='Drowsy %',
                   line=dict(color='#e74c3c', width=2)),
        secondary_y=True,
    )
    
    # Set titles
    fig.update_layout(
        title_text="Daily Session Metrics",
        font=dict(color=theme['font_color']),
        plot_bgcolor=theme['bgcolor'],
        paper_bgcolor=theme['bgcolor'],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=350,
    )
    
    # Set axes titles
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Average EAR", secondary_y=False)
    fig.update_yaxes(title_text="Drowsy %", secondary_y=True)
    
    return fig

def plot_realtime_gauge(avg_ear, avg_mar, efficiency_score):
    """Create gauge charts for current metrics"""
    theme = get_chart_theme()
    
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
    )
    
    # EAR Gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=avg_ear,
            title={"text": "Eye Openness (EAR)"},
            gauge={
                "axis": {"range": [0, 0.4], "tickwidth": 1},
                "bar": {"color": "#2ecc71" if avg_ear > 0.25 else "#e74c3c"},
                "steps": [
                    {"range": [0, 0.2], "color": "#e74c3c"},
                    {"range": [0.2, 0.25], "color": "#f39c12"},
                    {"range": [0.25, 0.4], "color": "#2ecc71"}
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": 0.25
                }
            },
            number={"valueformat": ".2f"}
        ),
        row=1, col=1
    )
    
    # MAR Gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=avg_mar,
            title={"text": "Mouth Openness (MAR)"},
            gauge={
                "axis": {"range": [0, 0.6], "tickwidth": 1},
                "bar": {"color": "#2ecc71" if avg_mar < 0.3 else "#e74c3c"},
                "steps": [
                    {"range": [0, 0.3], "color": "#2ecc71"},
                    {"range": [0.3, 0.4], "color": "#f39c12"},
                    {"range": [0.4, 0.6], "color": "#e74c3c"}
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": 0.3
                }
            },
            number={"valueformat": ".2f"}
        ),
        row=1, col=2
    )
    
    # Efficiency Score Gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=efficiency_score,
            title={"text": "Efficiency Score"},
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1},
                "bar": {"color": "#9b59b6"},
                "steps": [
                    {"range": [0, 0.3], "color": "#e74c3c"},
                    {"range": [0.3, 0.5], "color": "#f39c12"},
                    {"range": [0.5, 1], "color": "#2ecc71"}
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": 0.5
                }
            },
            number={"valueformat": ".2f"}
        ),
        row=1, col=3
    )
    
    fig.update_layout(
        height=250,
        font=dict(color=theme['font_color']),
        plot_bgcolor=theme['bgcolor'],
        paper_bgcolor=theme['bgcolor'],
    )
    
    return fig

def generate_alerts(df):
    """Generate alerts based on fatigue patterns"""
    alerts = []
    
    # Check for drowsy periods
    drowsy_periods = df[df['State'] == 'Drowsy']
    if len(drowsy_periods) > 10:
        alerts.append({
            "severity": "high",
            "message": f"⚠️ High fatigue detected: {len(drowsy_periods)} drowsy moments recorded."
        })
    
    # Check for yawning frequency
    yawning_periods = df[df['MAR'] > 0.4]
    if len(yawning_periods) > 5:
        alerts.append({
            "severity": "medium",
            "message": f"😴 Frequent yawning detected: {len(yawning_periods)} instances recorded."
        })
    
    # Check for low efficiency score
    low_efficiency = df[df['EfficiencyScore'] < 0.3]
    if len(low_efficiency) > len(df) * 0.3:  # More than 30% of time with low efficiency
        alerts.append({
            "severity": "high",
            "message": "📉 Low productivity alert: Extended periods of low efficiency detected."
        })
    
    return alerts

def display_alerts(alerts):
    """Display alerts in the UI"""
    if not alerts:
        st.success("✅ No alerts detected. Looking good!")
        return
    
    for alert in alerts:
        if alert["severity"] == "high":
            st.error(alert["message"])
        elif alert["severity"] == "medium":
            st.warning(alert["message"])
        else:
            st.info(alert["message"])

def render_settings_page():
    """Render the settings page"""
    st.header("⚙️ Dashboard Settings")
    
    st.subheader("Threshold Settings")
    with st.form("threshold_settings"):
        col1, col2 = st.columns(2)
        with col1:
            ear_threshold = st.slider("EAR Fatigue Threshold", 0.1, 0.4, 0.2, 0.01,
                                     help="Eye Aspect Ratio below this value indicates fatigue")
        with col2:
            mar_threshold = st.slider("MAR Yawning Threshold", 0.2, 0.6, 0.4, 0.01,
                                     help="Mouth Aspect Ratio above this value indicates yawning")
        
        efficiency_weight = st.slider("EAR Weight in Efficiency Score", 0.5, 0.9, 0.7, 0.05,
                                     help="Higher values give more importance to eye openness in efficiency calculation")
        
        submit = st.form_submit_button("Save Settings")
        if submit:
            # Save settings to a config file
            settings = {
                "ear_threshold": ear_threshold,
                "mar_threshold": mar_threshold,
                "efficiency_weight": efficiency_weight
            }
            with open("dashboard_settings.yaml", "w") as file:
                yaml.dump(settings, file)
            st.success("Settings saved successfully!")
    
    st.subheader("User Management")
    if st.session_state.get("role") == "admin":
        with st.expander("Add New User"):
            with st.form("add_user"):
                new_username = st.text_input("Username")
                new_password = st.text_input("Password", type="password")
                new_role = st.selectbox("Role", ["user", "admin"])
                
                submit = st.form_submit_button("Add User")
                if submit:
                    if not new_username or not new_password:
                        st.error("Username and password cannot be empty")
                    else:
                        success, message = register_user(new_username, new_password, new_role)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
        
        # Display existing users
        config = load_config()
        st.subheader("Existing Users")
        user_data = []
        for username, details in config["credentials"].items():
            user_data.append({
                "Username": username,
                "Role": details["role"]
            })
        
        st.table(pd.DataFrame(user_data))
    else:
        st.info("You need admin privileges to manage users.")
    
    st.subheader("Dashboard Appearance")
    st.radio("Theme", ["Dark", "Light"], key="theme", disabled=True,
            help="Theme selection is coming in a future update!")

def render_login_page():
    """Render the login page"""
    st.title("🔒 Employee Fatigue Dashboard")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if username and password:
                    authenticated, role = authenticate(username, password)
                    if authenticated:
                        st.session_state["authenticated"] = True
                        st.session_state["username"] = username
                        st.session_state["role"] = role
                        st.success("Login successful!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.warning("Please enter both username and password")
    
    with tab2:
        if not st.session_state.get("show_register", False):
            if st.button("Request Registration"):
                st.session_state["show_register"] = True
                st.info("Registration requests need admin approval. Please contact your administrator.")
        else:
            st.info("Registration requests need admin approval. Please contact your administrator.")
            st.button("Hide", on_click=lambda: st.session_state.update({"show_register": False}))

def render_main_dashboard(df):
    """Render the main dashboard UI"""
    # Dashboard header with title and description
    st.title("🧠 Advanced Employee Fatigue Dashboard")
    st.markdown("""
    Track eye and mouth metrics to monitor fatigue and productivity levels in remote work settings.
    This dashboard provides real-time analytics and personalized recommendations based on your fatigue patterns.
    """)
    
    # Sidebar for filters and navigation
    st.sidebar.title(f"👤 Welcome, {st.session_state.get('username', 'User')}")
    
    # Navigation
    page = st.sidebar.radio("Navigation", ["📊 Dashboard", "📈 Analytics", "🔍 Data Explorer", "⚙️ Settings"])
    
    # Time filter in sidebar
    st.sidebar.subheader("⏱️ Time Filter")
    time_filter = st.sidebar.selectbox("Select time range:", get_time_filter_options())
    
    if time_filter == "Custom Range":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start date", df['Timestamp'].min().date())
        with col2:
            end_date = st.date_input("End date", df['Timestamp'].max().date())
        filtered_df = apply_time_filter(df, time_filter, start_date, end_date)
    else:
        filtered_df = apply_time_filter(df, time_filter)
    
    # Session ID generator
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())[:8]
    
    st.sidebar.divider()
    st.sidebar.info(f"Session ID: {st.session_state['session_id']}")
    
    # Logout button in sidebar
    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.session_state["role"] = None
        st.rerun()
    
    # Handle different pages
    if page == "📊 Dashboard":
        render_dashboard_page(filtered_df)
    elif page == "📈 Analytics":
        render_analytics_page(filtered_df)
    elif page == "🔍 Data Explorer":
        render_data_explorer_page(filtered_df)
    elif page == "⚙️ Settings":
        render_settings_page()

def render_dashboard_page(df):
    """Render the main dashboard page with key metrics and charts"""
    # Calculate current metrics for gauges
    recent_data = df.iloc[-50:] if len(df) > 50 else df  # Get most recent data points
    avg_ear = recent_data['EAR'].mean()
    avg_mar = recent_data['MAR'].mean()
    efficiency_score = (avg_ear * 0.7) - (avg_mar * 0.3)
    
    # Top row: Realtime gauges
    st.subheader("📊 Current Metrics")
    st.plotly_chart(plot_realtime_gauge(avg_ear, avg_mar, efficiency_score), use_container_width=True)
    
    # Second row: Status and alerts
    st.subheader("⚠️ Status Alerts")
    alerts = generate_alerts(df)
    display_alerts(alerts)
    
    # Main metrics
    st.subheader("📊 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sessions = df['Date'].nunique()
        st.metric("Sessions", f"{total_sessions}")
    
    with col2:
        drowsy_count = len(df[df['State'] == 'Drowsy'])
        drowsy_percentage = (drowsy_count / len(df)) * 100
        st.metric("Drowsy %", f"{drowsy_percentage:.1f}%")
    
    with col3:
        efficiency_counts = df['EfficiencyLevel'].value_counts(normalize=True).to_dict()
        high_efficiency = efficiency_counts.get('High', 0) * 100
        st.metric("High Efficiency %", f"{high_efficiency:.1f}%")
    
    with col4:
        avg_efficiency = df['EfficiencyScore'].mean()
        st.metric("Avg Efficiency Score", f"{avg_efficiency:.2f}")
    
    # Trends chart
    st.subheader("📈 Fatigue Trends")
    
    # Create a smooth line chart with EAR and MAR
    fig = go.Figure()
    
    # Add EAR line
    fig.add_trace(go.Scatter(
        x=df['Timestamp'],
        y=df['EAR_Smoothed'],
        mode='lines',
        name='EAR (Eye Openness)',
        line=dict(color='#3498db', width=2)
    ))
    
    # Add MAR line
    fig.add_trace(go.Scatter(
        x=df['Timestamp'],
        y=df['MAR_Smoothed'],
        mode='lines',
        name='MAR (Mouth Openness)',
        line=dict(color='#e74c3c', width=2)
    ))
    
    # Add shaded areas for different states
    # Find consecutive drowsy periods
    drowsy_df = df[df['State'] == 'Drowsy']
    if not drowsy_df.empty:
        fig.add_trace(go.Scatter(
            x=drowsy_df['Timestamp'],
            y=drowsy_df['EAR_Smoothed'],
            mode='markers',
            marker=dict(color='red', size=8),
            name='Drowsy Periods'
        ))
    
    # Update layout
    theme = get_chart_theme()
    fig.update_layout(
        title='Eye and Mouth Metrics Over Time',
        xaxis_title='Time',
        yaxis_title='Ratio Value',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400,
        font=dict(color=theme['font_color']),
        plot_bgcolor=theme['bgcolor'],
        paper_bgcolor=theme['bgcolor'],
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show recommendations
    st.subheader("💡 Personalized Recommendations")
    recommendations = get_recommendations(df)
    
    for rec in recommendations:
        st.markdown(f"- {rec}")

def render_analytics_page(df):
    """Render the analytics page with detailed charts and insights"""
    st.header("📈 Advanced Analytics")
    
    # Tab layout for different analysis views
    tab1, tab2, tab3, tab4 = st.tabs(["Time Patterns", "Efficiency Analysis", "State Distribution", "Fatigue Heatmap"])
    
    with tab1:
        st.subheader("⏰ Time Analysis")
        
        # Hourly patterns
        if 'Hour' in df.columns:
            hourly_ear = df.groupby('Hour')['EAR'].mean().reset_index()
            hourly_ear = df.groupby('Hour')['EAR'].mean().reset_index()
            hourly_mar = df.groupby('Hour')['MAR'].mean().reset_index()
            
            # Create hourly pattern chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=hourly_ear['Hour'],
                y=hourly_ear['EAR'],
                mode='lines+markers',
                name='Avg EAR by Hour',
                line=dict(color='#3498db', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=hourly_mar['Hour'],
                y=hourly_mar['MAR'],
                mode='lines+markers',
                name='Avg MAR by Hour',
                line=dict(color='#e74c3c', width=2)
            ))
            
            theme = get_chart_theme()
            fig.update_layout(
                title='Hourly Fatigue Patterns',
                xaxis_title='Hour of Day',
                yaxis_title='Average Ratio',
                xaxis=dict(tickmode='linear', tick0=0, dtick=1),
                font=dict(color=theme['font_color']),
                plot_bgcolor=theme['bgcolor'],
                paper_bgcolor=theme['bgcolor'],
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add efficiency heatmap by hour
            st.plotly_chart(plot_hourly_heatmap(df), use_container_width=True)
        
        # Daily patterns
        st.subheader("📅 Daily Patterns")
        st.plotly_chart(plot_session_metrics(df), use_container_width=True)
    
    with tab2:
        st.subheader("🚀 Efficiency Analysis")
        
        # Distribution of efficiency scores
        fig = px.histogram(
            df, 
            x='EfficiencyScore',
            color='EfficiencyLevel',
            marginal='box',
            color_discrete_map={
                'Low': '#e74c3c',
                'Medium': '#f39c12',
                'High': '#2ecc71'
            },
            title='Distribution of Efficiency Scores'
        )
        
        theme = get_chart_theme()
        fig.update_layout(
            xaxis_title='Efficiency Score',
            yaxis_title='Count',
            font=dict(color=theme['font_color']),
            plot_bgcolor=theme['bgcolor'],
            paper_bgcolor=theme['bgcolor'],
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot of EAR vs MAR with efficiency coloring
        fig = px.scatter(
            df,
            x='EAR',
            y='MAR',
            color='EfficiencyLevel',
            color_discrete_map={
                'Low': '#e74c3c',
                'Medium': '#f39c12',
                'High': '#2ecc71'
            },
            opacity=0.7,
            title='EAR vs MAR Relationship by Efficiency Level'
        )
        
        fig.update_layout(
            xaxis_title='Eye Aspect Ratio (EAR)',
            yaxis_title='Mouth Aspect Ratio (MAR)',
            font=dict(color=theme['font_color']),
            plot_bgcolor=theme['bgcolor'],
            paper_bgcolor=theme['bgcolor'],
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Efficiency breakdown
        efficiency_counts = df['EfficiencyLevel'].value_counts().reset_index()
        efficiency_counts.columns = ['Level', 'Count']
        
        fig = px.pie(
            efficiency_counts,
            values='Count',
            names='Level',
            color='Level',
            color_discrete_map={
                'Low': '#e74c3c',
                'Medium': '#f39c12',
                'High': '#2ecc71'
            },
            title='Efficiency Level Breakdown'
        )
        
        fig.update_layout(
            font=dict(color=theme['font_color']),
            plot_bgcolor=theme['bgcolor'],
            paper_bgcolor=theme['bgcolor'],
            height=400
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("⭐ Key Insights")
            
            # Calculate some key metrics for insights
            high_efficiency_pct = (efficiency_counts[efficiency_counts['Level'] == 'High']['Count'].values[0] / efficiency_counts['Count'].sum()) * 100 if 'High' in efficiency_counts['Level'].values else 0
            
            st.markdown(f"""
            - **Overall Efficiency**: {high_efficiency_pct:.1f}% of time in high efficiency state
            - **Trend Analysis**: {'Positive trend, efficiency improving over sessions' if high_efficiency_pct > 50 else 'Opportunities for improvement in efficiency'}
            - **Recommendations**: Taking regular breaks can improve overall efficiency
            """)
            
            # Show hours of peak efficiency
            if 'Hour' in df.columns:
                hourly_efficiency = df.groupby('Hour')['EfficiencyScore'].mean().reset_index()
                if not hourly_efficiency.empty and 'EfficiencyScore' in hourly_efficiency.columns:
                    if hourly_efficiency['EfficiencyScore'].notna().any():
                        best_hour = hourly_efficiency.loc[hourly_efficiency['EfficiencyScore'].idxmax()]['Hour']
                        worst_hour = hourly_efficiency.loc[hourly_efficiency['EfficiencyScore'].idxmin()]['Hour']
                        st.markdown(f"""
                        - **Peak Efficiency**: Hour {int(best_hour)}:00 shows highest average efficiency  
                        - **Lowest Efficiency**: Hour {int(worst_hour)}:00 shows lowest average efficiency
                        """)
                    else:
                        st.info("EfficiencyScore column has only NaN values. Cannot determine peak hours.")
                else:
                    st.info("Insufficient data to calculate best and worst efficiency hours.")


                worst_hour = hourly_efficiency.loc[hourly_efficiency['EfficiencyScore'].idxmin()]['Hour']
                
                st.markdown(f"""
                - **Peak Efficiency**: Hour {int(best_hour)}:00 shows highest average efficiency
                - **Lowest Efficiency**: Hour {int(worst_hour)}:00 shows lowest average efficiency
                """)
    
    with tab3:
        st.subheader("🔍 State Distribution Analysis")
        
        # State breakdown
        state_counts = df['State'].value_counts().reset_index()
        state_counts.columns = ['State', 'Count']
        
        # Create a more interesting visualization with custom styling
        colors = {
            'Alert': '#2ecc71',
            'Drowsy': '#e74c3c',
            'Yawning': '#f39c12'
        }
        
        fig = go.Figure()
        
        for state in state_counts['State'].unique():
            value = state_counts[state_counts['State'] == state]['Count'].values[0]
            fig.add_trace(go.Bar(
                x=[state],
                y=[value],
                name=state,
                marker_color=colors.get(state, '#3498db'),
                text=[value],
                textposition='auto'
            ))
        
        theme = get_chart_theme()
        fig.update_layout(
            title='Distribution of States',
            xaxis_title='State',
            yaxis_title='Count',
            font=dict(color=theme['font_color']),
            plot_bgcolor=theme['bgcolor'],
            paper_bgcolor=theme['bgcolor'],
            height=400
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # State transitions analysis
            st.subheader("↔️ State Transitions")
            
            # Calculate state transitions
            df['PrevState'] = df['State'].shift(1)
            transitions = df.dropna(subset=['PrevState']).groupby(['PrevState', 'State']).size().reset_index()
            transitions.columns = ['From', 'To', 'Count']
            
            # Display transitions as a table
            st.write("Common State Transitions:")
            
            # Format the table with styled HTML
            transitions_html = """
            <div style="overflow-y: auto; max-height: 300px;">
            <table style="width:100%; border-collapse: collapse;">
                <tr style="background-color: #333; color: white;">
                    <th style="padding: 8px; text-align: left;">From State</th>
                    <th style="padding: 8px; text-align: left;">To State</th>
                    <th style="padding: 8px; text-align: left;">Count</th>
                </tr>
            """
            
            top_transitions = transitions.sort_values('Count', ascending=False).head(10)
            
            for _, row in top_transitions.iterrows():
                from_state = row['From']
                to_state = row['To']
                count = row['Count']
                
                bg_color = "#2c3e50"
                if from_state == 'Alert' and to_state == 'Drowsy':
                    bg_color = "#e74c3c22"  # Red with transparency
                elif from_state == 'Drowsy' and to_state == 'Alert':
                    bg_color = "#2ecc7122"  # Green with transparency
                
                transitions_html = """
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="background-color: #2c3e50;">
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">State 1</td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">State 2</td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">Count</td>
                </tr>
                <tr style="background-color: #2c3e50;">
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">Alert</td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">Drowsy</td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">42</td>
                </tr>
            </table>
            """

            
            
            st.markdown(transitions_html, unsafe_allow_html=True)
            
            # Calculate alertness recovery time
            recovery_times = []
            drowsy_start = None
            
            for i, row in df.iterrows():
                if row['State'] == 'Drowsy' and drowsy_start is None:
                    drowsy_start = row['Timestamp']
                elif row['State'] == 'Alert' and drowsy_start is not None:
                    recovery_time = (row['Timestamp'] - drowsy_start).total_seconds() / 60  # minutes
                    if 0 < recovery_time < 30:  # Filter out unrealistic values
                        recovery_times.append(recovery_time)
                    drowsy_start = None
            
            if recovery_times:
                avg_recovery = sum(recovery_times) / len(recovery_times)
                st.metric("Avg Recovery Time", f"{avg_recovery:.1f} min")
    
    with tab4:
        st.subheader("🔥 Fatigue Heatmap")
        
        # Create a heatmap of fatigue by hour and day
        st.plotly_chart(plot_hourly_heatmap(df), use_container_width=True)
        
        # Add a correlation matrix for metrics
        correlation_cols = ['EAR', 'MAR', 'EfficiencyScore', 'Blink_Rate', 'Yawn_Intensity', 'Activity']
        corr_matrix = df[correlation_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title='Correlation Matrix of Fatigue Metrics',
            aspect="auto"
        )
        
        theme = get_chart_theme()
        fig.update_layout(
            font=dict(color=theme['font_color']),
            plot_bgcolor=theme['bgcolor'],
            paper_bgcolor=theme['bgcolor'],
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add insights
        st.markdown("""
        ### Key Observations
        
        - Eye openness (EAR) tends to decrease during mid-afternoon hours (2-4 PM)
        - Mouth openness (MAR) shows higher activity during periods of low EAR
        - Time patterns are consistent across multiple sessions
        
        ### Recommendations
        
        - Schedule important tasks during high alertness hours
        - Take short breaks during identified fatigue periods
        - Consider adjustments to your work environment during low efficiency hours
        """)

def render_data_explorer_page(df):
    """Render the data explorer page with filtering and visualization options"""
    st.header("🔍 Data Explorer")
    
    # Check if min and max dates are the same
    min_date = df['Timestamp'].min().date()
    max_date = df['Timestamp'].max().date()
    
    # Handle case when min and max are identical
    if min_date == max_date:
        # Add a day to max_date to ensure they're different
        max_date = min_date + timedelta(days=1)
        
        # Show a note to the user
        st.info(f"Note: Data only contains records from {min_date}. Extending range to allow filtering.")
    
    # Now use the potentially adjusted date range
    date_range = st.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )
    
    # Rest of the function remains the same...
    
    start_date, end_date = date_range
    filtered_df = df[(df['Timestamp'].dt.date >= start_date) & (df['Timestamp'].dt.date <= end_date)]
    
    # Additional filters in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        state_filter = st.multiselect(
            "Filter by State",
            options=df['State'].unique(),
            default=df['State'].unique()
        )
    
    with col2:
        if 'FatigueLevel' in df.columns:
            fatigue_filter = st.multiselect(
                "Filter by Fatigue Level",
                options=df['FatigueLevel'].unique(),
                default=df['FatigueLevel'].unique()
            )
        else:
            fatigue_filter = None
    
    with col3:
        if 'Hour' in df.columns:
            hour_range = st.slider(
                "Hour of Day",
                min_value=0,
                max_value=23,
                value=(0, 23)
            )
        else:
            hour_range = (0, 23)
    
    # Apply filters
    if state_filter:
        filtered_df = filtered_df[filtered_df['State'].isin(state_filter)]
    
    if fatigue_filter:
        filtered_df = filtered_df[filtered_df['FatigueLevel'].isin(fatigue_filter)]
    
    if 'Hour' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['Hour'] >= hour_range[0]) & (filtered_df['Hour'] <= hour_range[1])]
    
    # Display filtered data stats
    st.subheader("📊 Filtered Data Statistics")
    st.write(f"Showing {len(filtered_df)} of {len(df)} records")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Data Table", "📊 Custom Visualization", "📥 Export"])
    
    with tab1:
        # Add quick search
        search_term = st.text_input("Quick Search (in any text column)")
        
        if search_term:
            # Search in string columns
            mask = pd.Series(False, index=filtered_df.index)
            for col in filtered_df.select_dtypes(include=['object']).columns:
                mask |= filtered_df[col].str.contains(search_term, case=False, na=False)
            
            filtered_df = filtered_df[mask]
        
        # Display paginated data
        page_size = st.slider("Records per page", 10, 100, 25)
        page_number = st.number_input("Page", min_value=1, max_value=max(1, len(filtered_df) // page_size + 1), value=1)
        
        start_idx = (page_number - 1) * page_size
        end_idx = min(start_idx + page_size, len(filtered_df))
        
        st.dataframe(filtered_df.iloc[start_idx:end_idx], use_container_width=True)
    
    with tab2:
        st.subheader("📊 Custom Visualization")
        
        # Let user select columns for visualization
        col1, col2 = st.columns(2)
        
        with col1:
            chart_type = st.selectbox(
                "Chart Type",
                ["Line Chart", "Scatter Plot", "Bar Chart", "Histogram", "Box Plot"]
            )
        
        with col2:
            if chart_type in ["Line Chart", "Bar Chart"]:
                groupby_col = st.selectbox(
                    "Group By",
                    options=[None, "Hour", "Date", "State", "FatigueLevel", "EfficiencyLevel"],
                    format_func=lambda x: x if x else "None"
                )
            else:
                groupby_col = None
        
        # Select columns for X and Y axes
        x_col = st.selectbox("X-axis", options=filtered_df.columns)
        
        if chart_type in ["Scatter Plot", "Line Chart"]:
            y_col = st.multiselect("Y-axis", options=filtered_df.columns, default=["EAR", "MAR"])
        else:
            y_col = st.selectbox("Y-axis", options=filtered_df.columns)
        
        # Color option
        color_col = st.selectbox(
            "Color By",
            options=[None, "State", "FatigueLevel", "EfficiencyLevel"],
            format_func=lambda x: x if x else "None"
        )
        
        # Generate chart based on selection
        if chart_type == "Line Chart":
            if groupby_col and groupby_col in filtered_df.columns:
                # Grouped line chart
                fig = px.line(
                    filtered_df.groupby(groupby_col)[y_col].mean().reset_index(),
                    x=groupby_col,
                    y=y_col,
                    color=color_col if color_col else None,
                    title=f"Average {y_col} by {groupby_col}"
                )
            else:
                # Regular line chart
                fig = px.line(
                    filtered_df,
                    x=x_col,
                    y=y_col,
                    color=color_col if color_col else None,
                    title=f"{y_col} over {x_col}"
                )
        
        elif chart_type == "Scatter Plot":
            fig = px.scatter(
                filtered_df,
                x=x_col,
                y=y_col[0] if y_col else None,
                color=color_col if color_col else None,
                opacity=0.7,
                title=f"{y_col[0] if y_col else ''} vs {x_col}"
            )
        
        elif chart_type == "Bar Chart":
            if groupby_col and groupby_col in filtered_df.columns:
                fig = px.bar(
                    filtered_df.groupby(groupby_col)[y_col].mean().reset_index(),
                    x=groupby_col,
                    y=y_col,
                    color=color_col if color_col else None,
                    title=f"Average {y_col} by {groupby_col}"
                )
            else:
                fig = px.bar(
                    filtered_df,
                    x=x_col,
                    y=y_col,
                    color=color_col if color_col else None,
                    title=f"{y_col} by {x_col}"
                )
        
        elif chart_type == "Histogram":
            fig = px.histogram(
                filtered_df,
                x=x_col,
                color=color_col if color_col else None,
                marginal="box",
                title=f"Distribution of {x_col}"
            )
        
        elif chart_type == "Box Plot":
            fig = px.box(
                filtered_df,
                x=x_col if color_col else None,
                y=y_col,
                color=color_col if color_col else None,
                title=f"Box Plot of {y_col}"
            )
        
        # Apply theme
        theme = get_chart_theme()
        fig.update_layout(
            font=dict(color=theme['font_color']),
            plot_bgcolor=theme['bgcolor'],
            paper_bgcolor=theme['bgcolor'],
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📥 Export Options")
        
        # Export format options
        export_format = st.radio(
            "Select Export Format",
            ["CSV", "Excel"]
        )
        
        # Export button
        if export_format == "CSV":
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download as CSV",
                csv,
                "fatigue_data_export.csv",
                "text/csv",
                key='download-csv'
            )
        else:
            # Create Excel file in memory
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name='Fatigue Data')
                
                # Access xlsxwriter workbook and worksheet objects
                workbook = writer.book
                worksheet = writer.sheets['Fatigue Data']
                
                # Add formats
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#0B5394',
                    'font_color': 'white',
                    'border': 1
                })
                
                # Apply header format
                for col_num, value in enumerate(filtered_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # Add a chart
                chart = workbook.add_chart({'type': 'line'})
                
                # Add series to chart
                chart.add_series({
                    'name': 'EAR',
                    'categories': ['Fatigue Data', 1, filtered_df.columns.get_loc('Timestamp'), len(filtered_df), filtered_df.columns.get_loc('Timestamp')],
                    'values': ['Fatigue Data', 1, filtered_df.columns.get_loc('EAR'), len(filtered_df), filtered_df.columns.get_loc('EAR')],
                })
                
                chart.add_series({
                    'name': 'MAR',
                    'categories': ['Fatigue Data', 1, filtered_df.columns.get_loc('Timestamp'), len(filtered_df), filtered_df.columns.get_loc('Timestamp')],
                    'values': ['Fatigue Data', 1, filtered_df.columns.get_loc('MAR'), len(filtered_df), filtered_df.columns.get_loc('MAR')],
                })
                
                chart.set_title({'name': 'EAR and MAR Trend'})
                chart.set_x_axis({'name': 'Time'})
                chart.set_y_axis({'name': 'Ratio'})
                
                # Insert chart
                worksheet.insert_chart('K2', chart, {'x_scale': 1.5, 'y_scale': 1.5})
            
            # Download button
            st.download_button(
                "📥 Download as Excel",
                output.getvalue(),
                "fatigue_data_export.xlsx",
                "application/vnd.ms-excel",
                key='download-excel'
            )
        
        # Report generation option
        st.subheader("📊 Generate Report")
        report_type = st.selectbox(
            "Report Type",
            ["Summary Report", "Detailed Analytics Report", "Custom Report"]
        )
        
        if st.button("Generate Report"):
            st.info("Generating report... Please wait.")
            time.sleep(1)  # Simulate report generation
            
            # Simple report in markdown
            st.markdown("### Fatigue Analysis Report")
            st.markdown(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown(f"**Data period:** {filtered_df['Timestamp'].min().strftime('%Y-%m-%d')} to {filtered_df['Timestamp'].max().strftime('%Y-%m-%d')}")
            st.markdown(f"**Total records:** {len(filtered_df)}")
            
            # Summary stats
            st.markdown("#### Summary Statistics")
            st.dataframe(filtered_df[['EAR', 'MAR', 'EfficiencyScore']].describe())
            
            # State breakdown
            st.markdown("#### State Distribution")
            state_counts = filtered_df['State'].value_counts().reset_index()
            state_counts.columns = ['State', 'Count']
            st.bar_chart(state_counts.set_index('State'))
            
            st.success("Report generated successfully!")

# -------------------- MAIN APP --------------------

def main():
    """Main application entry point"""
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    
    # Set page configuration
    st.set_page_config(
        page_title="Advanced Employee Fatigue Dashboard",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for styling
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stApp {
            background-color: #111;
            color: #f0f2f6;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 4px 4px 0px 0px;
            padding: 0px 20px;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(255, 255, 255, 0.1);
            border-bottom: 2px solid #3498db;
        }
        div[data-testid="stMetricValue"] {
            font-size: 2rem;
        }
        .styled-table {
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 0.9em;
            font-family: sans-serif;
            min-width: 400px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        }
        .styled-table thead tr {
            background-color: #3498db;
            color: #ffffff;
            text-align: left;
        }
        .styled-table th,
        .styled-table td {
            padding: 12px 15px;
        }
        .styled-table tbody tr {
            border-bottom: 1px solid #dddddd;
        }
        .styled-table tbody tr:nth-of-type(even) {
            background-color: rgba(255, 255, 255, 0.05);
        }
        .styled-table tbody tr:last-of-type {
            border-bottom: 2px solid #3498db;
        }
        .styled-table tbody tr.active-row {
            font-weight: bold;
            color: #3498db;
        }
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #192a56;
        }
        /* Button styling */
        .stButton > button {
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        .stButton > button:hover {
            background-color: #2980b9;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Check authentication
    if not st.session_state['authenticated']:
        render_login_page()
    else:
        # Load data
        df = load_data()
        # Render main dashboard
        render_main_dashboard(df)

if __name__ == "__main__":
    main()