from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report
)

# ------------------------------------------------------------
# Page setup and visual styling
# ------------------------------------------------------------
st.set_page_config(
    page_title='CNC Predictive Maintenance',
    page_icon='⚙️',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown('''
<style>
    :root { color-scheme: dark; }
    /* V2 palette: warm graphite + olive, intentionally not blue */
    .stApp { background: #171914; color: #eeeee5; }
    [data-testid="stSidebar"] { background: #20231b; border-right: 1px solid #3b4030; }
    [data-testid="stHeader"] { background: rgba(23,25,20,0.98); }
    [data-testid="stToolbar"] { background: transparent; }
    h1, h2, h3, h4 { color: #f4f2e9 !important; letter-spacing: -0.025em; }
    p, label, .stMarkdown { color: #d8d8ca; }
    [data-testid="stCaptionContainer"] { color: #a7aa98; }
    div[data-testid="stMetric"] { background: linear-gradient(145deg,#292d22,#22251d); border: 1px solid #424735; border-left: 4px solid #a7b85b; padding: 16px 18px; border-radius: 10px; }
    div[data-testid="stMetricLabel"] { color: #c0c2ad; }
    div[data-testid="stMetricValue"] { color: #f5f2e8; }
    div[data-testid="stMetricDelta"] { color: #b7ca70; }
    div[data-testid="stAlert"] { border-radius: 9px; }
    .hero { padding: 1.45rem 1.55rem; border-radius: 12px; background: linear-gradient(115deg,#303624,#25291f); border: 1px solid #4b5337; border-left: 5px solid #b3c568; margin-bottom: 1rem; }
    .hero-kicker { color: #c0d477; font-size: .76rem; font-weight: 750; letter-spacing: .14em; text-transform: uppercase; }
    .hero-title { color: #f7f4e9; font-size: 2rem; font-weight: 760; margin: .25rem 0; }
    .hero-sub { color: #d0d1c1; margin-bottom: .1rem; }
    .small-note { color: #a7aa98; font-size: .85rem; }
    .status-chip { display:inline-block; padding: .25rem .65rem; border-radius: 999px; font-weight: 700; font-size: .8rem; }
    .green { background:#354027; color:#d0e58a; }
    .red { background:#4a2b24; color:#ff9b83; }
    .amber { background:#494126; color:#f0d17b; }
    div.stButton > button, div[data-testid="stDownloadButton"] button { border-radius: 7px; font-weight: 650; border: 1px solid #555b42; }
    div.stButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] button { background: #879b43; border-color: #a7b85b; color: #fffdf2; }
    div.stButton > button:hover, div[data-testid="stDownloadButton"] button:hover { border-color: #c6d77c; color: #e0ecad; }
    [data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="textarea"] > div { background: #2b2e25; border-color: #4a4e3b; }
    [data-testid="stDataFrame"] { border: 1px solid #414634; border-radius: 8px; overflow: hidden; }
    hr { border-color: #414634; }
    [data-baseweb="radio"] label, [data-baseweb="checkbox"] label { color: #e4e3d8; }
    [data-testid="stProgressBar"] > div > div { background: #a7b85b; }
</style>
''', unsafe_allow_html=True)

DATA_FILENAME = 'cncmilingdata2023-2026.xlsx'
SHEET_NAME = 'Sensor_Data'
TARGET = 'Defect_Label'

NUMERIC_COLS = [
    'Spindle_Speed_RPM', 'Feed_Rate_mm_per_min', 'Cutting_Depth_mm',
    'Vibration_X_mm_s', 'Vibration_Y_mm_s', 'Vibration_Z_mm_s',
    'Spindle_Temp_C', 'Coolant_Temp_C', 'Coolant_Flow_L_per_min',
    'Power_Consumption_kW', 'Tool_Wear_Percent', 'Acoustic_Emission_dB',
    'Servo_Load_X_Percent', 'Servo_Load_Y_Percent', 'Servo_Load_Z_Percent',
    'Surface_Roughness_Ra_um'
]

THRESHOLDS = {
    'Vibration_X_mm_s': (3.0, 'high'),
    'Vibration_Y_mm_s': (3.0, 'high'),
    'Vibration_Z_mm_s': (2.5, 'high'),
    'Spindle_Temp_C': (85.0, 'high'),
    'Coolant_Temp_C': (55.0, 'high'),
    'Coolant_Flow_L_per_min': (2.0, 'low'),
    'Tool_Wear_Percent': (80.0, 'high'),
    'Acoustic_Emission_dB': (90.0, 'high'),
    'Power_Consumption_kW': (7.0, 'high'),
}

RECORDED_METRICS = pd.DataFrame([
    {'Model':'Random Forest (baseline)', 'Accuracy':97.18, 'Precision':97.56, 'Recall':75.00, 'F1-score':84.81, 'ROC-AUC':87.88},
    {'Model':'Decision Tree (baseline)', 'Accuracy':86.74, 'Precision':42.34, 'Recall':72.50, 'F1-score':53.46, 'ROC-AUC':84.93},
    {'Model':'Logistic Regression (threshold 0.25)', 'Accuracy':59.03, 'Precision':18.48, 'Recall':85.00, 'F1-score':30.36, 'ROC-AUC':86.93},
])

@st.cache_data(show_spinner=False)
def load_data(path_str):
    df = pd.read_excel(path_str, sheet_name=SHEET_NAME)
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.sort_values('Timestamp').reset_index(drop=True)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'Tool_Wear_Percent' in df.columns:
        df['Tool_Wear_Percent'] = df['Tool_Wear_Percent'].clip(0, 100)
    if 'Surface_Roughness_Ra_um' in df.columns:
        df['Surface_Roughness_Ra_um'] = df['Surface_Roughness_Ra_um'].clip(lower=0)
    return df


def build_features(df):
    work = df.drop(columns=['Timestamp', 'Failure_Mode'], errors='ignore').copy()
    work['Vibration_Magnitude'] = np.sqrt(
        work['Vibration_X_mm_s']**2 + work['Vibration_Y_mm_s']**2 + work['Vibration_Z_mm_s']**2
    )
    work['Temp_Difference'] = work['Spindle_Temp_C'] - work['Coolant_Temp_C']
    work['Efficiency_Ratio'] = work['Power_Consumption_kW'] / work['Spindle_Speed_RPM'].replace(0, np.nan)
    work['Spindle_Temp_Rolling'] = work['Spindle_Temp_C'].rolling(window=10, min_periods=1).mean()
    work['Vibration_Rolling'] = work['Vibration_Magnitude'].rolling(window=10, min_periods=1).mean()
    work['Temp_Rate_Change'] = work['Spindle_Temp_C'].diff().fillna(0)
    numeric = work.select_dtypes(include=[np.number]).columns
    work[numeric] = work[numeric].replace([np.inf, -np.inf], np.nan)
    work[numeric] = work[numeric].fillna(work[numeric].mean())
    return work

@st.cache_resource(show_spinner='Preparing prediction model...')
def get_prediction_engine(path_str, bundle_path_str):
    # Prefer the model bundle produced by the user's original pipeline.
    bundle_path = Path(bundle_path_str)
    if bundle_path.exists():
        try:
            bundle = joblib.load(bundle_path)
            if all(k in bundle for k in ['model', 'scaler', 'feature_columns']):
                return {'kind':'saved_bundle', 'model':bundle['model'], 'scaler':bundle['scaler'],
                        'features':bundle['feature_columns'], 'threshold':float(bundle.get('decision_threshold',0.5)),
                        'label':'Saved Logistic Regression bundle'}
        except Exception:
            pass
    # Fallback: train a Random Forest from the supplied dataset if the bundle is absent.
    df = load_data(path_str)
    features = build_features(df)
    if TARGET not in features:
        raise ValueError(f'The dataset must contain the target column {TARGET}.')
    X = features.drop(columns=[TARGET])
    y = features[TARGET].astype(int)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=160, random_state=42, class_weight='balanced_subsample', n_jobs=-1)
    model.fit(X_train, y_train)
    return {'kind':'fallback_rf', 'model':model, 'scaler':None, 'features':list(X.columns), 'threshold':0.5,
            'label':'Fallback Random Forest (trained when dashboard opened)'}


def prepare_one_reading(reading, feature_cols):
    one = pd.DataFrame([reading])
    one['Vibration_Magnitude'] = np.sqrt(one['Vibration_X_mm_s']**2 + one['Vibration_Y_mm_s']**2 + one['Vibration_Z_mm_s']**2)
    one['Temp_Difference'] = one['Spindle_Temp_C'] - one['Coolant_Temp_C']
    one['Efficiency_Ratio'] = one['Power_Consumption_kW'] / one['Spindle_Speed_RPM'].replace(0, np.nan)
    # A single live reading has no historical window; use current values as the rolling proxies.
    one['Spindle_Temp_Rolling'] = one['Spindle_Temp_C']
    one['Vibration_Rolling'] = one['Vibration_Magnitude']
    one['Temp_Rate_Change'] = 0.0
    one = one.reindex(columns=feature_cols)
    return one.replace([np.inf, -np.inf], np.nan).fillna(0)


def infer(engine, reading):
    x = prepare_one_reading(reading, engine['features'])
    if engine['scaler'] is not None:
        x_input = engine['scaler'].transform(x)
    else:
        x_input = x
    model = engine['model']
    if hasattr(model, 'predict_proba'):
        prob = float(model.predict_proba(x_input)[0][1])
    else:
        score = float(model.decision_function(x_input)[0])
        prob = 1 / (1 + np.exp(-score))
    threshold = engine['threshold']
    pred = int(prob >= threshold)
    return pred, prob, threshold


def get_alerts(reading):
    alerts = []
    for sensor, (limit, direction) in THRESHOLDS.items():
        value = float(reading.get(sensor, 0))
        if direction == 'low' and value < limit:
            alerts.append({'Sensor':sensor, 'Reading':value, 'Condition':f'Below {limit:g}', 'Severity':'High', 'Recommendation':'Check coolant pump, filter and coolant lines.'})
        elif direction == 'high' and value > limit:
            recommendations = {
                'Spindle_Temp_C':'Inspect spindle cooling and lubrication.',
                'Coolant_Temp_C':'Inspect coolant temperature and heat-exchanger performance.',
                'Tool_Wear_Percent':'Inspect and consider replacing the cutting tool.',
                'Acoustic_Emission_dB':'Check for chatter, tool damage and unusual contact.',
                'Power_Consumption_kW':'Inspect spindle load and cutting conditions.',
                'Vibration_X_mm_s':'Inspect bearings, spindle and machine alignment.',
                'Vibration_Y_mm_s':'Inspect bearings, spindle and machine alignment.',
                'Vibration_Z_mm_s':'Inspect bearings, spindle and machine alignment.'
            }
            alerts.append({'Sensor':sensor, 'Reading':value, 'Condition':f'Above {limit:g}', 'Severity':'High', 'Recommendation':recommendations.get(sensor,'Inspect the sensor and machine condition.')})
    return alerts


def failure_mode(reading, probability, threshold):
    if reading['Spindle_Temp_C'] > 85 and reading['Coolant_Flow_L_per_min'] < 2:
        return 'Thermal overload'
    if reading['Tool_Wear_Percent'] > 95:
        return 'Tool wear failure'
    if reading['Tool_Wear_Percent'] > 80:
        return 'Tool wear warning'
    if reading['Coolant_Flow_L_per_min'] < 1:
        return 'Coolant system failure'
    if reading['Coolant_Flow_L_per_min'] < 2:
        return 'Coolant flow warning'
    if any(reading[v] > 1.0 for v in ['Vibration_X_mm_s','Vibration_Y_mm_s','Vibration_Z_mm_s']):
        return 'Vibration anomaly'
    if reading['Power_Consumption_kW'] > 7:
        return 'Spindle load anomaly'
    if probability >= threshold:
        return 'Anomalous operating pattern'
    return 'Normal operation'


def recommendations(reading):
    items = []
    if reading['Spindle_Temp_C'] > 85: items.append('Inspect spindle cooling system and lubrication.')
    if reading['Coolant_Flow_L_per_min'] < 2: items.append('Check coolant pump, filter and coolant lines.')
    if reading['Tool_Wear_Percent'] > 80: items.append('Inspect tool condition; replace if wear limit is exceeded.')
    if reading['Power_Consumption_kW'] > 7: items.append('Inspect spindle load, feed rate and cutting conditions.')
    if reading['Acoustic_Emission_dB'] > 90: items.append('Check for chatter, tool damage and abnormal contact.')
    if any(reading[v] > 1.0 for v in ['Vibration_X_mm_s','Vibration_Y_mm_s','Vibration_Z_mm_s']): items.append('Inspect bearings, spindle alignment and vibration sources.')
    if not items: items.append('No rule-based issue detected; continue routine monitoring.')
    return items


def get_path(filename):
    candidates = [Path.cwd()/filename, Path(__file__).resolve().parent/filename, Path('/mnt/data')/filename, Path('/mnt/data/files')/filename]
    for p in candidates:
        if p.exists(): return p
    return None

# ------------------------------------------------------------
# Load dataset / model
# ------------------------------------------------------------
data_path = get_path(DATA_FILENAME)
if data_path is None:
    st.error(f'Could not find `{DATA_FILENAME}`. Put the Excel dataset in the same folder as this dashboard.')
    st.stop()

try:
    df = load_data(str(data_path))
    feature_frame = build_features(df)
    engine = get_prediction_engine(str(data_path), str(data_path.parent/'model_bundle.joblib'))
except Exception as exc:
    st.error(f'Could not prepare the dataset/model: {exc}')
    st.info('Check that the Excel file contains the `Sensor_Data` sheet and the sensor columns used by the project.')
    st.stop()

# ------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------
st.sidebar.markdown('## ⚙️ CNC MONITOR')
st.sidebar.caption('GRAPHITE / OLIVE THEME · V2')
st.sidebar.caption('Predictive maintenance · ML prototype')
page = st.sidebar.radio('NAVIGATION', ['Overview', 'Live Prediction', 'Dataset Explorer', 'Model Performance', 'Alerts & Reports'])
st.sidebar.divider()
st.sidebar.markdown('**Project**')
st.sidebar.caption('Intelligent Predictive Maintenance System for CNC Milling Machines')
st.sidebar.caption(f'Data source: `{data_path.name}`')
st.sidebar.caption(f'Prediction engine: {engine["label"]}')
st.sidebar.caption('Prototype only — validate thresholds and models before industrial deployment.')

# Common values
n_rows = len(df)
normal_count = int((df[TARGET] == 0).sum()) if TARGET in df else 0
defect_count = int((df[TARGET] == 1).sum()) if TARGET in df else 0
defect_pct = (100 * defect_count / n_rows) if n_rows else 0

if page == 'Overview':
    st.markdown('''<div class="hero"><div class="hero-kicker">Industrial intelligence · Machine health</div><div class="hero-title">CNC Predictive Maintenance</div><div class="hero-sub">Monitor sensor conditions, explore defect patterns and support maintenance decisions with machine-learning predictions.</div></div>''', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Sensor records', f'{n_rows:,}')
    c2.metric('Normal records', f'{normal_count:,}')
    c3.metric('Defective records', f'{defect_count:,}')
    c4.metric('Recorded defect rate', f'{defect_pct:.1f}%')
    left,right = st.columns([1.15,1])
    with left:
        st.subheader('Dataset class distribution')
        dist = pd.DataFrame({'Condition':['Normal','Defective'], 'Records':[normal_count, defect_count]})
        fig = px.pie(dist, names='Condition', values='Records', hole=.62, color='Condition', color_discrete_map={'Normal':'#9fb85b','Defective':'#e9785d'})
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend_title_text='')
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader('Sensor signals at a glance')
        sensor_choice = st.selectbox('Choose sensor', ['Spindle_Temp_C','Vibration_X_mm_s','Tool_Wear_Percent','Power_Consumption_kW','Coolant_Flow_L_per_min','Acoustic_Emission_dB'])
        plot_df = df[[sensor_choice, TARGET]].dropna().copy()
        plot_df['Condition'] = plot_df[TARGET].map({0:'Normal',1:'Defective'})
        fig = px.histogram(plot_df, x=sensor_choice, color='Condition', barmode='overlay', opacity=.72,
                           color_discrete_map={'Normal':'#9fb85b','Defective':'#e9785d'})
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend_title_text='')
        st.plotly_chart(fig, use_container_width=True)
    st.subheader('Recorded model results from your original run')
    st.dataframe(RECORDED_METRICS.style.format({c:'{:.2f}%' for c in ['Accuracy','Precision','Recall','F1-score','ROC-AUC']}), use_container_width=True, hide_index=True)
    st.caption('These are the recorded experiment results from the terminal output you provided. They are not recalculated every time the dashboard opens.')

elif page == 'Live Prediction':
    st.title('Live Sensor Prediction')
    st.write('Enter a machine reading to estimate defect probability and generate rule-based maintenance alerts.')
    defaults = {
        'Spindle_Speed_RPM':3010.0,'Feed_Rate_mm_per_min':498.0,'Cutting_Depth_mm':2.5,
        'Vibration_X_mm_s':0.52,'Vibration_Y_mm_s':0.48,'Vibration_Z_mm_s':0.31,
        'Spindle_Temp_C':44.0,'Coolant_Temp_C':25.5,'Coolant_Flow_L_per_min':8.1,
        'Power_Consumption_kW':3.6,'Tool_Wear_Percent':22.0,'Acoustic_Emission_dB':61.0,
        'Servo_Load_X_Percent':40.0,'Servo_Load_Y_Percent':38.5,'Servo_Load_Z_Percent':35.0,
        'Surface_Roughness_Ra_um':1.62
    }
    with st.form('prediction_form'):
        st.markdown('#### Operating parameters')
        cols = st.columns(4)
        values = {}
        ranges = {
            'Spindle_Speed_RPM':(0,8000),'Feed_Rate_mm_per_min':(0,2000),'Cutting_Depth_mm':(0,20),
            'Vibration_X_mm_s':(0,20),'Vibration_Y_mm_s':(0,20),'Vibration_Z_mm_s':(0,20),
            'Spindle_Temp_C':(-20,200),'Coolant_Temp_C':(-20,120),'Coolant_Flow_L_per_min':(0,50),
            'Power_Consumption_kW':(0,100),'Tool_Wear_Percent':(0,100),'Acoustic_Emission_dB':(0,160),
            'Servo_Load_X_Percent':(0,200),'Servo_Load_Y_Percent':(0,200),'Servo_Load_Z_Percent':(0,200),
            'Surface_Roughness_Ra_um':(0,100)
        }
        for i, (key, default) in enumerate(defaults.items()):
            with cols[i%4]:
                values[key] = st.number_input(key.replace('_',' '), min_value=float(ranges[key][0]), max_value=float(ranges[key][1]), value=float(default), step=0.1, format='%.3f', key=f'input_{key}')
        submitted = st.form_submit_button('Analyse machine reading', type='primary', use_container_width=True)
    if submitted:
        pred, prob, threshold = infer(engine, values)
        alerts = get_alerts(values)
        mode = failure_mode(values, prob, threshold)
        if prob >= threshold:
            st.error(f'POTENTIAL DEFECT DETECTED · Estimated model probability {prob:.1%}')
        else:
            st.success(f'MODEL CLASSIFICATION: NORMAL · Estimated model probability {prob:.1%}')
        k1,k2,k3 = st.columns(3)
        k1.metric('Defect probability', f'{prob:.1%}')
        k2.metric('Decision threshold', f'{threshold:.3f}')
        k3.metric('Rule-based alerts', len(alerts))
        st.progress(min(max(prob,0.0),1.0), text='Model defect probability')
        st.markdown(f'**Rule-based failure-mode label:** `{mode}`')
        if alerts:
            st.subheader('Threshold alerts')
            st.dataframe(pd.DataFrame(alerts), use_container_width=True, hide_index=True)
        else:
            st.info('No rule-based sensor thresholds were crossed for this reading.')
        st.subheader('Recommended actions')
        for action in recommendations(values): st.markdown(f'- {action}')
        st.caption('The model probability and rule-based failure label are separate signals. These outputs are advisory and require engineering validation.')

elif page == 'Dataset Explorer':
    st.title('Dataset Explorer')
    st.write('Filter and inspect the source records. Use the export button to download the filtered rows.')
    filtered = df.copy()
    if 'Timestamp' in filtered.columns:
        valid_dates = filtered['Timestamp'].dropna()
        if len(valid_dates):
            date_range = st.date_input('Timestamp range', value=(valid_dates.min().date(), valid_dates.max().date()))
            if isinstance(date_range, tuple) and len(date_range)==2:
                filtered = filtered[(filtered['Timestamp'].dt.date >= date_range[0]) & (filtered['Timestamp'].dt.date <= date_range[1])]
    condition = st.selectbox('Condition', ['All', 'Normal only', 'Defective only'])
    if condition == 'Normal only': filtered = filtered[filtered[TARGET] == 0]
    elif condition == 'Defective only': filtered = filtered[filtered[TARGET] == 1]
    search_col = st.selectbox('Sensor to summarise', NUMERIC_COLS)
    a,b,c = st.columns(3)
    a.metric('Rows after filters', f'{len(filtered):,}')
    b.metric(f'Mean {search_col.replace("_"," ")}', f'{filtered[search_col].mean():.2f}' if len(filtered) else '—')
    c.metric(f'Max {search_col.replace("_"," ")}', f'{filtered[search_col].max():.2f}' if len(filtered) else '—')
    st.dataframe(filtered, use_container_width=True, height=420)
    st.download_button('Download filtered data as CSV', filtered.to_csv(index=False).encode('utf-8'), file_name='cnc_filtered_sensor_data.csv', mime='text/csv')

elif page == 'Model Performance':
    st.title('Model Performance')
    st.info('This page reports the metrics captured in your original run. The Logistic Regression confusion matrix uses the tuned 0.25 decision threshold.')
    st.dataframe(RECORDED_METRICS.style.format({c:'{:.2f}%' for c in ['Accuracy','Precision','Recall','F1-score','ROC-AUC']}), use_container_width=True, hide_index=True)
    st.subheader('Logistic Regression confusion matrix — recorded run')
    cm = np.array([[763,600],[24,136]])
    fig = go.Figure(data=go.Heatmap(z=cm, x=['Predicted Normal','Predicted Defective'], y=['Actual Normal','Actual Defective'],
                                    text=cm, texttemplate='%{text}', colorscale=[[0,'#f0eedf'],[0.35,'#c2cd87'],[1,'#687b32']], showscale=True))
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title='Predicted', yaxis_title='Actual')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('''**Interpretation**
- **763 true negatives:** normal readings correctly identified.
- **600 false positives:** normal readings incorrectly flagged as defective.
- **24 false negatives:** defective readings missed by the model.
- **136 true positives:** defective readings correctly detected.''')
    st.warning('The 0.25 threshold prioritised recall (85.0%) but generated many false positives. The threshold was selected using the test labels in the recorded script, so this is an experimental result rather than an unbiased final validation. Tune thresholds on a validation set for a formal evaluation.')

elif page == 'Alerts & Reports':
    st.title('Batch Scan & Maintenance Report')
    st.write('Scan a subset of dataset readings, then export the model predictions and threshold alerts as a CSV report.')
    max_rows = min(2000, len(df))
    count = st.slider('Number of records to scan', min_value=min(25,max_rows), max_value=max_rows, value=min(250,max_rows), step=max(1,min(25,max_rows//20)))
    seed = st.number_input('Sampling seed', min_value=0, max_value=99999, value=42, step=1)
    if st.button('Run batch scan', type='primary'):
        scan = df.sample(n=int(count), random_state=int(seed)).copy() if len(df) > count else df.copy()
        feature_data = build_features(scan)
        records = []
        progress = st.progress(0, text='Analysing readings...')
        for i, (_, row) in enumerate(scan.iterrows()):
            reading = {col: float(row[col]) if pd.notna(row[col]) else float(df[col].median()) for col in NUMERIC_COLS}
            pred, prob, threshold = infer(engine, reading)
            alerts = get_alerts(reading)
            records.append({
                'Timestamp': row['Timestamp'] if 'Timestamp' in scan.columns else None,
                'Actual_Defect_Label': int(row[TARGET]) if TARGET in scan.columns and pd.notna(row[TARGET]) else None,
                'Predicted_Defect': pred,
                'Defect_Probability_Percent': round(prob*100,2),
                'Decision_Threshold': round(threshold,3),
                'Failure_Mode': failure_mode(reading,prob,threshold),
                'Alert_Count': len(alerts),
                'Alert_Sensors': ', '.join(a['Sensor'] for a in alerts),
                'Recommended_Actions': ' | '.join(recommendations(reading))
            })
            if i % max(1, int(count)//50) == 0:
                progress.progress(min((i+1)/int(count),1.0), text=f'Analysing reading {i+1:,} of {int(count):,}')
        report = pd.DataFrame(records)
        progress.progress(1.0, text='Batch scan complete')
        st.session_state['batch_report'] = report
    if 'batch_report' in st.session_state:
        report = st.session_state['batch_report']
        c1,c2,c3 = st.columns(3)
        c1.metric('Scanned records', f'{len(report):,}')
        c2.metric('Predicted defective', f'{int(report.Predicted_Defect.sum()):,}')
        c3.metric('Records with threshold alerts', f'{int((report.Alert_Count>0).sum()):,}')
        st.dataframe(report, use_container_width=True, height=420)
        st.download_button('Download maintenance report CSV', report.to_csv(index=False).encode('utf-8'), file_name='cnc_predictive_maintenance_report.csv', mime='text/csv')

st.sidebar.divider()
st.sidebar.caption('Built as an internship demonstration dashboard · CNC ML prototype')
