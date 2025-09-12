import dash
from dash import dcc, html, Input, Output, callback, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import base64
import io

warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(df=None):
    """Load and preprocess the Titanic dataset or a user-uploaded dataset"""
    if df is None:
        # Load default dataset
        df = sns.load_dataset('titanic')
    
    # Handle missing values
    if 'age' in df.columns:
        df['age'].fillna(df['age'].median(), inplace=True)
    if 'fare' in df.columns:
        df['fare'].fillna(df['fare'].median(), inplace=True)
    if 'embarked' in df.columns:
        df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
    
    # Create additional features
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], 
                                bins=[0, 12, 18, 35, 60, 100], 
                                labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
    
    if 'fare' in df.columns:
        df['fare_category'] = pd.cut(df['fare'], 
                                    bins=[0, 10, 30, 100, 1000], 
                                    labels=['Low', 'Medium', 'High', 'Premium'])
    
    if 'sibsp' in df.columns and 'parch' in df.columns:
        df['family_size'] = df['sibsp'] + df['parch'] + 1
        df['family_category'] = pd.cut(df['family_size'], 
                                      bins=[0, 1, 4, 20], 
                                      labels=['Alone', 'Small Family', 'Large Family'])
    
    if 'survived' in df.columns:
        df['survival_status'] = df['survived'].map({0: 'Did Not Survive', 1: 'Survived'})
    if 'sex' in df.columns:
        df['gender'] = df['sex'].str.title()
    
    return df

# Load initial data
df = load_and_preprocess_data()

# ============================================================================
# DASHBOARD STYLING
# ============================================================================

COLORS = {
    'primary': '#4A90E2',
    'secondary': '#7B68EE',
    'accent': '#50C878',
    'danger': '#FF6B6B',
    'warning': '#FFD93D',
    'background': '#F8F9FA',
    'card': '#FFFFFF',
    'text': '#2C3E50',
    'border': '#E1E8ED'
}

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# ============================================================================
# DASH APP INITIALIZATION
# ============================================================================

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Titanic Data Analysis Dashboard"

# ============================================================================
# DASHBOARD LAYOUT
# ============================================================================

app.layout = html.Div([
    dcc.Store(id='stored-data'),
    
    # Header Section
    html.Div([
        html.H1("GENERIC DATA VISUALIZATION DASHBOARD", style={'textAlign': 'center'}),
        html.P("Upload a dataset to generate visualizations automatically", style={'textAlign': 'center'}),
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        ),
        html.Div(id='output-data-upload'),
    ], style={'backgroundColor': COLORS['background'], 'padding': '20px'}),
    
    # Control Panel
    html.Div(id='control-panel-container', style={'display': 'none'}, children=[
        html.H4("Create a Chart"),
        html.Div([
            html.Div([
                html.Label("Chart Type"),
                dcc.Dropdown(
                    id='chart-type-dropdown',
                    options=[
                        {'label': 'Bar Chart', 'value': 'bar'},
                        {'label': 'Scatter Plot', 'value': 'scatter'},
                        {'label': 'Histogram', 'value': 'histogram'},
                        {'label': 'Pie Chart', 'value': 'pie'},
                        {'label': 'Box Plot', 'value': 'box'},
                    ]
                )
            ], className='three columns'),
            html.Div([
                html.Label("X-Axis"),
                dcc.Dropdown(id='x-axis-dropdown')
            ], className='three columns'),
            html.Div([
                html.Label("Y-Axis"),
                dcc.Dropdown(id='y-axis-dropdown')
            ], className='three columns'),
            html.Div([
                html.Label("Color"),
                dcc.Dropdown(id='color-dropdown')
            ], className='three columns'),
        ], className='row'),
        html.Button('Add Chart', id='add-chart-button', n_clicks=0)
    ]),
    
    # Charts
    html.Div(id='charts-container', className='row'),
    
    # Footer
    html.Div([
        html.P("© 2024 Generic Data Visualization Dashboard",
               style={'textAlign': 'center', 'margin': '20px 0'})
    ], style={'backgroundColor': COLORS['background'], 'padding': '10px'})
    
], style={'backgroundColor': COLORS['background'], 'minHeight': '100vh'})

# ============================================================================
# CALLBACKS
# ============================================================================

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return html.Div(['Please upload a CSV or Excel file.'])
    except Exception as e:
        return html.Div([f'There was an error processing this file: {e}'])
    return df

@app.callback(
    Output('stored-data', 'data'),
    Output('control-panel-container', 'style'),
    Output('x-axis-dropdown', 'options'),
    Output('y-axis-dropdown', 'options'),
    Output('color-dropdown', 'options'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_store(contents, filename):
    if contents is not None:
        df = parse_contents(contents, filename)
        if isinstance(df, pd.DataFrame):
            options = [{'label': col, 'value': col} for col in df.columns]
            return df.to_json(date_format='iso', orient='split'), {'display': 'block'}, options, options, options
    return None, {'display': 'none'}, [], [], []

@app.callback(Output('output-data-upload', 'children'),
              Input('stored-data', 'data'))
def update_output(data):
    if data is not None:
        return html.Div(['Data uploaded and processed successfully!'])
    return html.Div(['Using preloaded Titanic dataset.'])

def get_df(data):
    if data is not None:
        return pd.read_json(data, orient='split')
    return load_and_preprocess_data()

@app.callback(
    Output('charts-container', 'children'),
    Input('add-chart-button', 'n_clicks'),
    State('charts-container', 'children'),
    State('stored-data', 'data'),
    State('chart-type-dropdown', 'value'),
    State('x-axis-dropdown', 'value'),
    State('y-axis-dropdown', 'value'),
    State('color-dropdown', 'value')
)
def add_chart(n_clicks, existing_children, data, chart_type, x_axis, y_axis, color):
    if n_clicks == 0 or not data:
        return []
    
    df = get_df(data)
    fig = go.Figure()
    
    if chart_type == 'bar':
        fig = px.bar(df, x=x_axis, y=y_axis, color=color, title=f'Bar Chart of {y_axis} by {x_axis}')
    elif chart_type == 'scatter':
        fig = px.scatter(df, x=x_axis, y=y_axis, color=color, title=f'Scatter Plot of {y_axis} vs {x_axis}')
    elif chart_type == 'histogram':
        fig = px.histogram(df, x=x_axis, color=color, title=f'Histogram of {x_axis}')
    elif chart_type == 'pie':
        fig = px.pie(df, names=x_axis, values=y_axis, title=f'Pie Chart of {x_axis}')
    elif chart_type == 'box':
        fig = px.box(df, x=x_axis, y=y_axis, color=color, title=f'Box Plot of {y_axis} by {x_axis}')
        
    new_chart = html.Div([dcc.Graph(figure=fig)], className='six columns')
    
    if existing_children is None:
        existing_children = []
        
    existing_children.append(new_chart)
    return existing_children

# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == '__main__':
    print("Starting Unified Dashboard...")
    print("Dashboard will be available at: http://127.0.0.1:8050/")
    print("Open the link in your browser to view the dashboard")
    print("Press Ctrl+C to stop the server")
    
    app.run(debug=True, host='127.0.0.1', port=8050)
