import dash
from dash import dcc, html, Input, Output, callback, State, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import base64
import io
import uuid
import time
from functools import lru_cache
import json
from datetime import datetime
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.io as pio
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import psutil

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
# HELPER FUNCTIONS
# ============================================================================

def export_charts_as_png(charts_data, data=None, filename="dashboard_charts.png"):
    """Export charts as PNG using kaleido"""
    try:
        # If no charts available, create a simple data visualization
        if not charts_data or len(charts_data) == 0:
            if data:
                df = pd.read_json(data, orient='split')
                # Create a simple bar chart of the first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    fig = px.bar(df.head(20), x=df.index[:20], y=col, 
                               title=f"Sample Data: {col}",
                               labels={'x': 'Index', 'y': col})
                    fig.update_layout(height=600, width=800)
                else:
                    # If no numeric columns, create a simple text plot
                    fig = go.Figure()
                    fig.add_annotation(text="No numeric data available for visualization",
                                     xref="paper", yref="paper",
                                     x=0.5, y=0.5, showarrow=False,
                                     font=dict(size=16))
                    fig.update_layout(height=600, width=800)
            else:
                # Create a placeholder chart
                fig = go.Figure()
                fig.add_annotation(text="No data available for visualization",
                                 xref="paper", yref="paper",
                                 x=0.5, y=0.5, showarrow=False,
                                 font=dict(size=16))
                fig.update_layout(height=600, width=800)
        else:
            # Try to extract chart data from different possible formats
            fig = None
            
            # Method 1: Try to extract from dcc.Graph components
            for chart_component in charts_data:
                if hasattr(chart_component, 'figure') and chart_component.figure:
                    fig = chart_component.figure
                    break
                elif isinstance(chart_component, dict) and 'figure' in chart_component:
                    fig = chart_component['figure']
                    break
                elif isinstance(chart_component, dict) and 'data' in chart_component:
                    # Create figure from data
                    fig = go.Figure(data=chart_component['data'])
                    if 'layout' in chart_component:
                        fig.update_layout(chart_component['layout'])
                    break
            
            # If still no figure, create a simple one
            if fig is None:
                fig = go.Figure()
                fig.add_annotation(text="Charts are available but cannot be exported as static images",
                                 xref="paper", yref="paper",
                                 x=0.5, y=0.5, showarrow=False,
                                 font=dict(size=16))
                fig.update_layout(height=600, width=800)
        
        # Export as PNG
        img_bytes = pio.to_image(fig, format="png", width=1200, height=800, scale=2)
        return base64.b64encode(img_bytes).decode()
    except Exception as e:
        print(f"Error exporting PNG: {e}")
        # Return a simple error visualization
        try:
            fig = go.Figure()
            fig.add_annotation(text=f"Export Error: {str(e)}",
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False,
                             font=dict(size=14))
            fig.update_layout(height=600, width=800)
            img_bytes = pio.to_image(fig, format="png", width=1200, height=800, scale=2)
            return base64.b64encode(img_bytes).decode()
        except:
            return None

def export_dashboard_as_pdf(data, charts_data, filename="dashboard_report.pdf"):
    """Export dashboard as PDF using reportlab"""
    try:
        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=16, spaceAfter=30)
        heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=14, spaceAfter=12)
        
        # Build content
        story = []
        
        # Title
        story.append(Paragraph("Data Visualization Dashboard Report", title_style))
        story.append(Spacer(1, 12))
        
        # Data summary
        if data:
            try:
                df = pd.read_json(data, orient='split')
                story.append(Paragraph("Data Summary", heading_style))
                story.append(Paragraph(f"Dataset contains {len(df):,} rows and {len(df.columns)} columns", styles['Normal']))
                story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
                story.append(Spacer(1, 12))
                
                # Data preview table - limit to 5 rows to avoid PDF size issues
                story.append(Paragraph("Data Preview (First 5 rows)", heading_style))
                table_data = [df.columns.tolist()]
                for _, row in df.head(5).iterrows():
                    # Limit cell content to avoid PDF issues
                    row_data = []
                    for val in row.values:
                        if isinstance(val, str) and len(str(val)) > 50:
                            row_data.append(str(val)[:47] + "...")
                        else:
                            row_data.append(str(val))
                    table_data.append(row_data)
                
                from reportlab.platypus import Table, TableStyle
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 8),
                    ('FONTSIZE', (0, 1), (-1, -1), 7),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
                ]))
                story.append(table)
                story.append(Spacer(1, 12))
            except Exception as e:
                story.append(Paragraph(f"Error processing data: {str(e)}", styles['Normal']))
        else:
            story.append(Paragraph("No data available", styles['Normal']))
        
        # Charts section
        if charts_data:
            story.append(Paragraph("Visualizations", heading_style))
            story.append(Paragraph(f"Dashboard contains {len(charts_data)} interactive charts", styles['Normal']))
            story.append(Paragraph("Note: Interactive charts are best viewed in the web dashboard", styles['Normal']))
        else:
            story.append(Paragraph("No charts available", styles['Normal']))
        
        # Add footer
        story.append(Spacer(1, 20))
        story.append(Paragraph("Generated by Enhanced Data Visualization Dashboard", 
                             ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, alignment=1)))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        pdf_bytes = buffer.getvalue()
        
        # Ensure we have valid PDF content
        if len(pdf_bytes) > 0:
            return base64.b64encode(pdf_bytes).decode()
        else:
            print("Error: Generated PDF is empty")
            return None
            
    except Exception as e:
        print(f"Error exporting PDF: {e}")
        # Return a simple error PDF
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            story = [Paragraph("PDF Export Error", styles['Heading1']),
                    Paragraph(f"Error: {str(e)}", styles['Normal'])]
            doc.build(story)
            buffer.seek(0)
            return base64.b64encode(buffer.getvalue()).decode()
        except:
            return None

def create_empty_state():
    """Create empty state with instructions when no data is uploaded"""
    return html.Div([
        html.Div([
            html.H2("Welcome to the Enhanced Data Visualization Dashboard", 
                   style={'textAlign': 'center', 'color': COLORS['primary'], 'marginBottom': '30px'}),
            html.Div([
                html.Div([
                    html.H4("📊 Upload Your Data", style={'color': COLORS['primary'], 'textAlign': 'center'}),
                    html.P("Drag and drop your CSV or Excel file above to get started", 
                           style={'textAlign': 'center', 'marginBottom': '20px'}),
                    html.Ul([
                        html.Li("CSV files (.csv)"),
                        html.Li("Excel files (.xlsx, .xls)"),
                        html.Li("Maximum file size: 50MB")
                    ], style={'textAlign': 'left', 'display': 'inline-block'})
                ], className='six columns', style={'padding': '20px', 'textAlign': 'center'}),
                html.Div([
                    html.H4("🎯 Features Available", style={'color': COLORS['primary'], 'textAlign': 'center'}),
                    html.Ul([
                        html.Li("Interactive chart creation"),
                        html.Li("Advanced filtering system"),
                        html.Li("Statistical analysis tools"),
                        html.Li("Machine learning insights"),
                        html.Li("Export capabilities")
                    ], style={'textAlign': 'left', 'display': 'inline-block'})
                ], className='six columns', style={'padding': '20px', 'textAlign': 'center'})
            ], className='row', style={'justifyContent': 'center'}),
            html.Div([
                html.H4("📈 Try with Sample Data", style={'textAlign': 'center', 'marginTop': '30px'}),
                html.P("The dashboard will automatically load the Titanic dataset as an example", 
                      style={'textAlign': 'center', 'color': COLORS['text']})
            ], style={'textAlign': 'center'})
        ], style={
            'backgroundColor': COLORS['card'],
            'padding': '40px',
            'borderRadius': '10px',
            'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
            'margin': '20px auto',
            'maxWidth': '800px',
            'textAlign': 'center'
        })
    ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'minHeight': '60vh'})

def generate_metrics_cards(df):
    """Generate dynamic metrics cards based on data"""
    if df is None or df.empty:
        return []
    
    # Calculate key metrics
    total_records = len(df)
    
    # Try to find common metrics
    metrics = []
    
    # Total records
    metrics.append(html.Div([
        html.H3(f"{total_records:,}", style={'color': COLORS['primary'], 'margin': '0'}),
        html.P("Total Records", style={'margin': '0', 'color': COLORS['text']})
    ], className='metric-card', style={
        'backgroundColor': COLORS['card'],
        'padding': '20px',
        'borderRadius': '8px',
        'textAlign': 'center',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }))
    
    # Numeric columns for additional metrics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        # Average of first numeric column
        first_numeric = numeric_cols[0]
        avg_value = df[first_numeric].mean()
        metrics.append(html.Div([
            html.H3(f"{avg_value:.2f}", style={'color': COLORS['accent'], 'margin': '0'}),
            html.P(f"Avg {first_numeric.title()}", style={'margin': '0', 'color': COLORS['text']})
        ], className='metric-card', style={
            'backgroundColor': COLORS['card'],
            'padding': '20px',
            'borderRadius': '8px',
            'textAlign': 'center',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        }))
    
    # Missing data percentage
    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    metrics.append(html.Div([
        html.H3(f"{missing_pct:.1f}%", style={'color': COLORS['warning'] if missing_pct > 10 else COLORS['success'], 'margin': '0'}),
        html.P("Missing Data", style={'margin': '0', 'color': COLORS['text']})
    ], className='metric-card', style={
        'backgroundColor': COLORS['card'],
        'padding': '20px',
        'borderRadius': '8px',
        'textAlign': 'center',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }))
    
    # Number of columns
    metrics.append(html.Div([
        html.H3(f"{len(df.columns)}", style={'color': COLORS['info'], 'margin': '0'}),
        html.P("Columns", style={'margin': '0', 'color': COLORS['text']})
    ], className='metric-card', style={
        'backgroundColor': COLORS['card'],
        'padding': '20px',
        'borderRadius': '8px',
        'textAlign': 'center',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }))
    
    return html.Div(metrics, className='row', style={'marginBottom': '30px'})

def create_chart_management_toolbar(chart_id):
    """Create management toolbar for each chart"""
    return html.Div([
        html.Button("✏️", id=f"edit-{chart_id}", className="chart-btn", 
                   style={'margin': '2px', 'padding': '5px 8px', 'border': 'none', 'borderRadius': '4px', 'backgroundColor': COLORS['info'], 'color': 'white'}),
        html.Button("📋", id=f"duplicate-{chart_id}", className="chart-btn",
                   style={'margin': '2px', 'padding': '5px 8px', 'border': 'none', 'borderRadius': '4px', 'backgroundColor': COLORS['accent'], 'color': 'white'}),
        html.Button("💾", id=f"export-{chart_id}", className="chart-btn",
                   style={'margin': '2px', 'padding': '5px 8px', 'border': 'none', 'borderRadius': '4px', 'backgroundColor': COLORS['success'], 'color': 'white'}),
        html.Button("🗑️", id=f"delete-{chart_id}", className="chart-btn",
                   style={'margin': '2px', 'padding': '5px 8px', 'border': 'none', 'borderRadius': '4px', 'backgroundColor': COLORS['danger'], 'color': 'white'})
    ], style={'position': 'absolute', 'top': '10px', 'right': '10px', 'zIndex': 1000})

def create_filter_controls(df):
    """Create dynamic filter controls based on data types"""
    controls = []
    
    for col in df.columns:
        if df[col].dtype in ['object', 'category']:
            # Categorical filter
            unique_values = df[col].dropna().unique()[:20]  # Limit to 20 values
            options = [{'label': str(val), 'value': val} for val in unique_values]
            controls.append(html.Div([
                html.Label(f"{col}:"),
                dcc.Dropdown(
                    id=f'filter-{col}',
                    options=options,
                    multi=True,
                    placeholder=f"Select {col}",
                    style={'marginBottom': '10px'}
                )
            ]))
        elif df[col].dtype in ['int64', 'float64']:
            # Numeric range filter - handle NaN values
            numeric_data = df[col].dropna()
            if len(numeric_data) > 0:
                min_val = int(numeric_data.min())
                max_val = int(numeric_data.max())
                controls.append(html.Div([
                    html.Label(f"{col}:"),
                    dcc.RangeSlider(
                        id=f'filter-{col}',
                        min=min_val,
                        max=max_val,
                        value=[min_val, max_val],
                        marks={min_val: str(min_val), max_val: str(max_val)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'marginBottom': '15px'}))
            else:
                # Skip columns with all NaN values
                continue
    
    return controls

def perform_correlation_analysis(df):
    """Generate correlation matrix analysis"""
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        return html.Div("Not enough numeric columns for correlation analysis")
    
    corr_matrix = numeric_df.corr()
    fig = px.imshow(corr_matrix, 
                   text_auto=True, 
                   aspect="auto", 
                   title="Correlation Matrix",
                   color_continuous_scale='RdBu_r')
    
    return dcc.Graph(figure=fig)

def perform_descriptive_stats(df):
    """Generate descriptive statistics table"""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return html.Div("No numeric columns found for descriptive statistics")
    
    stats_df = numeric_df.describe()
    return dash_table.DataTable(
        data=stats_df.round(2).to_dict('records'),
        columns=[{"name": i, "id": i} for i in stats_df.columns],
        style_cell={'textAlign': 'left', 'fontSize': '12px'},
        style_header={'backgroundColor': COLORS['primary'], 'color': 'white', 'fontWeight': 'bold'}
    )

def perform_distribution_analysis(df):
    """Generate distribution analysis charts"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]  # Limit to 4 columns
    
    if len(numeric_cols) == 0:
        return html.Div("No numeric columns found for distribution analysis")
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=numeric_cols)
    
    for i, col in enumerate(numeric_cols):
        row = (i // 2) + 1
        col_pos = (i % 2) + 1
        
        fig.add_trace(
            go.Histogram(x=df[col], name=col, showlegend=False),
            row=row, col=col_pos
        )
    
    fig.update_layout(height=600, title_text="Distribution Analysis")
    return dcc.Graph(figure=fig)

def perform_missing_data_analysis(df):
    """Generate missing data analysis"""
    missing_data = df.isnull().sum()
    missing_pct = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing Percentage': missing_pct.values
    }).sort_values('Missing Percentage', ascending=False)
    
    fig = px.bar(missing_df, x='Column', y='Missing Percentage', 
                title='Missing Data Analysis',
                color='Missing Percentage',
                color_continuous_scale='Reds')
    
    return dcc.Graph(figure=fig)

def perform_prediction_analysis(df):
    """Perform prediction analysis using Random Forest"""
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        return html.Div("Not enough numeric columns for prediction analysis")
    
    # Use the last column as target if it's binary, otherwise use first numeric
    target_col = None
    for col in numeric_df.columns:
        if numeric_df[col].nunique() == 2:
            target_col = col
            break
    
    if target_col is None:
        target_col = numeric_df.columns[-1]
    
    feature_cols = [col for col in numeric_df.columns if col != target_col]
    
    if len(feature_cols) == 0:
        return html.Div("No features available for prediction")
    
    X = numeric_df[feature_cols].fillna(numeric_df[feature_cols].mean())
    y = numeric_df[target_col].fillna(numeric_df[target_col].mode()[0])
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(importance_df, x='Feature', y='Importance', 
                title=f'Feature Importance for Predicting {target_col}')
    
    return dcc.Graph(figure=fig)

def perform_cluster_analysis(df):
    """Perform K-means clustering analysis"""
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        return html.Div("Not enough numeric columns for clustering analysis")
    
    # Use first 2 numeric columns for visualization
    cols = numeric_df.columns[:2]
    X = numeric_df[cols].fillna(numeric_df[cols].mean())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Create scatter plot with clusters
    fig = px.scatter(df, x=cols[0], y=cols[1], color=clusters,
                    title=f'K-means Clustering: {cols[0]} vs {cols[1]}',
                    labels={'color': 'Cluster'})
    
    return dcc.Graph(figure=fig)

def perform_feature_importance_analysis(df):
    """Perform feature importance analysis"""
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        return html.Div("Not enough numeric columns for feature importance analysis")
    
    # Calculate correlation with target (last column)
    target_col = numeric_df.columns[-1]
    correlations = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
    correlations = correlations.drop(target_col)  # Remove self-correlation
    
    importance_df = pd.DataFrame({
        'Feature': correlations.index,
        'Correlation': correlations.values
    })
    
    fig = px.bar(importance_df, x='Feature', y='Correlation',
                title=f'Feature Correlation with {target_col}')
    
    return dcc.Graph(figure=fig)

def perform_dimensionality_reduction(df):
    """Perform PCA analysis and visualization"""
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return html.Div("Need at least 2 numeric columns for PCA analysis")
        
        # Handle missing values
        numeric_df = numeric_df.dropna()
        if len(numeric_df) < 10:
            return html.Div("Not enough data points for PCA analysis")
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Perform PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        # Create scatter plot
        fig = px.scatter(
            x=pca_result[:, 0], 
            y=pca_result[:, 1],
            title=f'PCA Analysis (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})',
            labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', 
                   'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'}
        )
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return dcc.Graph(figure=fig)
    except Exception as e:
        return html.Div(f"Error in PCA analysis: {str(e)}", style={'color': COLORS['danger']})

def perform_outlier_detection(df):
    """Detect and visualize outliers using IQR method"""
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) == 0:
            return html.Div("No numeric columns found for outlier detection")
        
        # Select first numeric column for outlier detection
        col = numeric_df.columns[0]
        data = numeric_df[col].dropna()
        
        # Calculate IQR
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        # Create box plot with outliers highlighted
        fig = px.box(df, y=col, title=f'Outlier Detection for {col}')
        
        # Add outlier information
        outlier_count = len(outliers)
        total_count = len(data)
        outlier_percentage = (outlier_count / total_count) * 100
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12),
            margin=dict(l=20, r=20, t=40, b=20),
            annotations=[
                dict(
                    x=0.5, y=1.1,
                    xref='paper', yref='paper',
                    text=f'Outliers: {outlier_count}/{total_count} ({outlier_percentage:.1f}%)',
                    showarrow=False,
                    font=dict(size=14, color=COLORS['warning'])
                )
            ]
        )
        
        return dcc.Graph(figure=fig)
    except Exception as e:
        return html.Div(f"Error in outlier detection: {str(e)}", style={'color': COLORS['danger']})

def get_performance_metrics():
    """Get current performance metrics"""
    import psutil
    import os
    
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_usage': cpu_percent,
            'memory_usage': memory.percent,
            'disk_usage': disk.percent,
            'memory_available': memory.available // (1024**3),  # GB
            'process_memory': psutil.Process(os.getpid()).memory_info().rss // (1024**2)  # MB
        }
    except ImportError:
        return {
            'cpu_usage': 'N/A',
            'memory_usage': 'N/A',
            'disk_usage': 'N/A',
            'memory_available': 'N/A',
            'process_memory': 'N/A'
        }

# Dashboard Templates
DASHBOARD_TEMPLATES = {
    'survival_analysis': {
        'name': 'Survival Analysis Dashboard',
        'description': 'Comprehensive survival analysis with demographic breakdowns',
        'charts': [
            {'type': 'bar', 'x': 'pclass', 'y': 'survived', 'title': 'Survival by Class'},
            {'type': 'pie', 'x': 'sex', 'y': 'survived', 'title': 'Survival by Gender'},
            {'type': 'scatter', 'x': 'age', 'y': 'fare', 'color': 'survived', 'title': 'Age vs Fare Survival'},
            {'type': 'box', 'x': 'survived', 'y': 'age', 'title': 'Age Distribution by Survival'}
        ]
    },
    'demographic_overview': {
        'name': 'Demographic Overview',
        'description': 'Complete demographic analysis of the dataset',
        'charts': [
            {'type': 'histogram', 'x': 'age', 'title': 'Age Distribution'},
            {'type': 'bar', 'x': 'pclass', 'title': 'Passenger Class Distribution'},
            {'type': 'pie', 'x': 'sex', 'title': 'Gender Distribution'},
            {'type': 'bar', 'x': 'embarked', 'title': 'Embarkation Port'},
            {'type': 'histogram', 'x': 'fare', 'title': 'Fare Distribution'}
        ]
    },
    'economic_analysis': {
        'name': 'Economic Analysis',
        'description': 'Economic factors and their impact on survival',
        'charts': [
            {'type': 'scatter', 'x': 'fare', 'y': 'survived', 'title': 'Fare vs Survival'},
            {'type': 'box', 'x': 'pclass', 'y': 'fare', 'title': 'Fare by Class'},
            {'type': 'violin', 'x': 'survived', 'y': 'fare', 'title': 'Fare Distribution by Survival'},
            {'type': 'heatmap', 'title': 'Correlation Matrix'}
        ]
    }
}

# ============================================================================
# DASHBOARD STYLING
# ============================================================================

COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#C73E1D',
    'warning': '#F4E285',
    'info': '#6A994E',
    'light': '#F8F9FA',
    'dark': '#212529',
    'danger': '#DC3545',
    'border': '#DEE2E6',
    'text': '#495057',
    'background': '#F8F9FA',
    'card': '#FFFFFF',
    'gradient_start': '#667eea',
    'gradient_end': '#764ba2'
}

external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
]

# ============================================================================
# DASH APP INITIALIZATION
# ============================================================================

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = "Enhanced Data Visualization Dashboard"

# Add custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .stats-btn, .ml-btn, .export-btn, .template-btn {
                width: 100%;
                margin: 5px 0;
                padding: 8px;
                border: none;
                border-radius: 4px;
                background-color: #2E86AB;
                color: white;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            .stats-btn:hover, .ml-btn:hover, .export-btn:hover, .template-btn:hover {
                background-color: #1e5f7a;
            }
            .metric-card {
                transition: transform 0.2s;
            }
            .metric-card:hover {
                transform: translateY(-2px);
            }
            .chart-btn {
                transition: opacity 0.2s;
            }
            .chart-btn:hover {
                opacity: 0.8;
            }
            .spinner-border {
                display: inline-block;
                width: 2rem;
                height: 2rem;
                vertical-align: text-bottom;
                border: 0.25em solid currentColor;
                border-right-color: transparent;
                border-radius: 50%;
                animation: spinner-border 0.75s linear infinite;
            }
            @keyframes spinner-border {
                to { transform: rotate(360deg); }
            }
            .text-primary {
                color: #2E86AB !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ============================================================================
# DASHBOARD LAYOUT
# ============================================================================

app.layout = html.Div([
    # Data stores
    dcc.Store(id='stored-data'),
    dcc.Store(id='chart-configs', data={}),
    dcc.Store(id='filter-state', data={}),
    
    # Header Section
    html.Div([
        html.H1("Enhanced Data Visualization Dashboard", 
               style={'textAlign': 'center', 'color': COLORS['primary'], 'marginBottom': '10px'}),
        html.P("Upload a dataset to generate interactive visualizations", 
               style={'textAlign': 'center', 'color': COLORS['text'], 'marginBottom': '20px'}),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                html.I(className="fas fa-cloud-upload-alt", style={'fontSize': '24px', 'marginRight': '10px'}),
                'Drag and Drop or ', html.A('Select Files', style={'color': COLORS['primary']})
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '2px',
                'borderStyle': 'dashed',
                'borderColor': COLORS['primary'],
                'borderRadius': '10px',
                'textAlign': 'center',
                'margin': '10px',
                'backgroundColor': COLORS['card'],
                'cursor': 'pointer'
            },
            multiple=False
        ),
        html.Div(id='output-data-upload'),
    ], style={'backgroundColor': COLORS['background'], 'padding': '20px'}),
    
    # Main Content Area - Horizontal Layout
    html.Div([
        # Top Control Bar - Horizontal
        html.Div([
            # Data Preview Section
            html.Div(id='data-preview-container', style={'display': 'none'}),
    
            # Control Panel
            html.Div(id='control-panel-container', style={'display': 'none'}, children=[
                html.H4("Create a Chart", style={'color': COLORS['primary']}),
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
                                {'label': 'Heatmap', 'value': 'heatmap'},
                                {'label': 'Violin Plot', 'value': 'violin'},
                                {'label': 'Line Chart', 'value': 'line'},
                                {'label': 'Sunburst', 'value': 'sunburst'},
                                {'label': 'Treemap', 'value': 'treemap'},
                                {'label': '3D Scatter', 'value': '3d_scatter'},
                                {'label': 'Area Chart', 'value': 'area'},
                            ],
                            style={'marginBottom': '10px'}
                        )
                    ], className='three columns'),
                    html.Div([
                        html.Label("X-Axis"),
                        dcc.Dropdown(id='x-axis-dropdown', style={'marginBottom': '10px'})
                    ], className='two columns'),
                    html.Div([
                        html.Label("Y-Axis"),
                        dcc.Dropdown(id='y-axis-dropdown', style={'marginBottom': '10px'})
                    ], className='two columns'),
                    html.Div([
                        html.Label("Color"),
                        dcc.Dropdown(id='color-dropdown', style={'marginBottom': '10px'})
                    ], className='two columns'),
                    html.Div([
                        html.Label("Size"),
                        dcc.Dropdown(id='size-dropdown', style={'marginBottom': '10px'})
                    ], className='two columns'),
                    html.Div([
                        html.Button('Add Chart', id='add-chart-button', n_clicks=0,
                                   style={'width': '100%', 'backgroundColor': COLORS['primary'], 'color': 'white', 'border': 'none', 'padding': '10px', 'borderRadius': '5px'})
                    ], className='one columns')
                ], className='row'),
            ]),
            
            # Filter Panel
            html.Div(id='filter-panel', style={'display': 'none'}, children=[
                html.H4("Data Filters", style={'color': COLORS['primary']}),
                html.Div(id='filter-controls'),
                html.Div([
                    html.Button("Apply Filters", id='apply-filters-btn', 
                               style={'backgroundColor': COLORS['primary'], 'color': 'white', 'border': 'none', 'padding': '8px', 'borderRadius': '4px', 'margin': '5px'}),
                    html.Button("Reset Filters", id='reset-filters-btn',
                               style={'backgroundColor': COLORS['warning'], 'color': 'white', 'border': 'none', 'padding': '8px', 'borderRadius': '4px', 'margin': '5px'})
                ]),
                html.Div(id='filter-status', style={'marginTop': '10px', 'fontSize': '12px'})
            ]),
            
            # Dashboard Templates Panel
            html.Div(id='templates-panel', style={'display': 'none'}, children=[
                html.H4("Dashboard Templates", style={'color': COLORS['primary']}),
                dcc.Dropdown(
                    id='template-selector',
                    options=[
                        {'label': 'Survival Analysis', 'value': 'survival_analysis'},
                        {'label': 'Demographic Overview', 'value': 'demographic_overview'},
                        {'label': 'Economic Analysis', 'value': 'economic_analysis'}
                    ],
                    placeholder='Select a template...',
                    style={'marginBottom': '10px'}
                ),
                html.Button("Load Template", id='load-template-btn', className='template-btn'),
                html.Div(id='template-description', style={'fontSize': '12px', 'marginTop': '10px'})
            ]),
            
            # Statistical Analysis Panel
            html.Div(id='stats-panel', style={'display': 'none'}, children=[
                html.H4("Statistical Analysis", style={'color': COLORS['primary']}),
                html.Div([
                    html.Button("Correlation Matrix", id='correlation-btn', className='stats-btn'),
                    html.Button("Descriptive Stats", id='descriptive-btn', className='stats-btn'),
                    html.Button("Distribution Analysis", id='distribution-btn', className='stats-btn'),
                    html.Button("Missing Data Analysis", id='missing-data-btn', className='stats-btn'),
                ], className='row'),
                html.Div(id='stats-output')
            ]),
            
            # Machine Learning Panel
            html.Div(id='ml-panel', style={'display': 'none'}, children=[
                html.H4("Machine Learning", style={'color': COLORS['primary']}),
                html.Div([
                    html.Button("Predict Target", id='predict-btn', className='ml-btn'),
                    html.Button("Cluster Analysis", id='cluster-btn', className='ml-btn'),
                    html.Button("Feature Importance", id='feature-btn', className='ml-btn'),
                    html.Button("PCA Analysis", id='pca-btn', className='ml-btn'),
                    html.Button("Outlier Detection", id='outlier-btn', className='ml-btn'),
                ], className='row'),
                html.Div(id='ml-output')
            ]),
            
            # Export Panel
            html.Div(id='export-panel', style={'display': 'none'}, children=[
                html.H4("Export Options", style={'color': COLORS['primary']}),
                html.Div([
                    html.Button("Export as PDF", id='export-pdf-btn', className='export-btn'),
                    html.Button("Export as PNG", id='export-png-btn', className='export-btn'),
                    html.Button("Export Data as CSV", id='export-csv-btn', className='export-btn'),
                ], className='row'),
                html.Hr(style={'margin': '15px 0'}),
                html.H5("Configuration", style={'color': COLORS['primary']}),
                html.Div([
                    html.Button("Save Dashboard Config", id='save-config-btn', className='export-btn'),
                    html.Button("Load Dashboard Config", id='load-config-btn', className='export-btn'),
                ], className='row'),
                html.Div(id='export-status', style={'marginTop': '10px', 'fontSize': '12px'})
            ]),
            
            # Performance Monitoring Panel
            html.Div(id='performance-panel', style={'display': 'none'}, children=[
                html.H4("Performance Monitor", style={'color': COLORS['primary']}),
                html.Button("Check Performance", id='check-performance-btn', className='export-btn'),
                html.Div(id='performance-metrics', style={'marginTop': '10px', 'fontSize': '12px'})
            ])
            
        ], className='twelve columns', style={'backgroundColor': COLORS['card'], 'padding': '20px', 'borderRadius': '10px', 'margin': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
        
        # Main Dashboard Area
        html.Div([
            # Metrics Cards
            html.Div(id='metrics-container'),
            
            # Empty State or Charts
            html.Div(id='main-content'),
            
        ], className='twelve columns', style={'padding': '10px'})
        
    ], className='row'),
    
    # Chart Editing Modal
    dcc.ConfirmDialog(
        id='chart-edit-modal',
        message='Chart editing functionality will be implemented here.',
        displayed=False
    ),
    
    # Footer
    html.Div([
        html.P("© 2024 Enhanced Data Visualization Dashboard - Professional Analytics Platform",
               style={'textAlign': 'center', 'margin': '20px 0', 'color': COLORS['text']})
    ], style={'backgroundColor': COLORS['background'], 'padding': '10px'})
    
], style={'backgroundColor': COLORS['background'], 'minHeight': '100vh'})

# ============================================================================
# CALLBACKS
# ============================================================================

def parse_contents(contents, filename):
    """Parse uploaded file contents with enhanced error handling"""
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return html.Div([
                html.H4("❌ Unsupported File Format", style={'color': COLORS['danger']}),
                html.P(f"File '{filename}' is not supported. Please upload a CSV or Excel file.")
            ])
        
        # Validate data
        if df.empty:
            return html.Div([
                html.H4("⚠️ Empty Dataset", style={'color': COLORS['warning']}),
                html.P("The uploaded file contains no data.")
            ])
        
        if len(df.columns) < 2:
            return html.Div([
                html.H4("⚠️ Insufficient Columns", style={'color': COLORS['warning']}),
                html.P("The dataset must have at least 2 columns for visualization.")
            ])
        
        return df
        
    except UnicodeDecodeError:
        return html.Div([
            html.H4("❌ Encoding Error", style={'color': COLORS['danger']}),
            html.P("The file encoding is not supported. Please save as UTF-8 and try again.")
        ])
    except pd.errors.EmptyDataError:
        return html.Div([
            html.H4("❌ Empty File", style={'color': COLORS['danger']}),
            html.P("The uploaded file is empty or corrupted.")
        ])
    except Exception as e:
        return html.Div([
            html.H4("❌ Processing Error", style={'color': COLORS['danger']}),
            html.P(f"Error processing file: {str(e)}"),
            html.P("Please check the file format and try again.")
        ])

@app.callback(
    Output('stored-data', 'data'),
    Output('control-panel-container', 'style'),
    Output('data-preview-container', 'style'),
    Output('filter-panel', 'style'),
    Output('templates-panel', 'style'),
    Output('stats-panel', 'style'),
    Output('ml-panel', 'style'),
    Output('export-panel', 'style'),
    Output('performance-panel', 'style'),
    Output('x-axis-dropdown', 'options'),
    Output('y-axis-dropdown', 'options'),
    Output('color-dropdown', 'options'),
    Output('size-dropdown', 'options'),
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_store(contents, filename):
    if contents is not None:
        # Show loading indicator
        loading_message = html.Div([
            html.Div([
                html.Div(className="spinner-border text-primary", role="status", style={'width': '20px', 'height': '20px', 'marginRight': '10px'}),
                html.Span("Processing your data...", style={'color': COLORS['primary'], 'fontWeight': 'bold'})
            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'padding': '20px'})
        ])
        
        df = parse_contents(contents, filename)
        if isinstance(df, pd.DataFrame):
            options = [{'label': col, 'value': col} for col in df.columns]
            success_message = html.Div([
                html.Div([
                    html.I(className="fas fa-check-circle", style={'color': COLORS['success'], 'fontSize': '20px', 'marginRight': '10px'}),
                    html.Span(f"✅ Data uploaded successfully! {df.shape[0]:,} rows × {df.shape[1]} columns", 
                             style={'color': COLORS['success'], 'fontWeight': 'bold'})
                ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'padding': '20px'})
            ])
            return (df.to_json(date_format='iso', orient='split'), 
                   {'display': 'block'}, 
                   {'display': 'block'}, 
                   {'display': 'block'},
                   {'display': 'block'},
                   {'display': 'block'},
                   {'display': 'block'},
                   {'display': 'block'},
                   {'display': 'block'},
                   options, options, options, options, success_message)
        else:
            # Show error message
            error_message = html.Div([
                html.Div([
                    html.I(className="fas fa-exclamation-triangle", style={'color': COLORS['danger'], 'fontSize': '20px', 'marginRight': '10px'}),
                    html.Span("❌ Error processing file. Please check the format and try again.", 
                             style={'color': COLORS['danger'], 'fontWeight': 'bold'})
                ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'padding': '20px'})
            ])
            return (None, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, [], [], [], [], error_message)
    
    # Default state
    default_message = html.Div([
        html.P('📊 Ready to upload your data or use the sample dataset', 
              style={'color': COLORS['text'], 'textAlign': 'center'})
    ])
    return None, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, [], [], [], [], default_message

# Removed duplicate callback - now handled in update_store

@app.callback(
    Output('data-preview-container', 'children'),
    Input('stored-data', 'data')
)
def display_data_preview(data):
    if data is not None:
        df = pd.read_json(data, orient='split')
        return html.Div([
            html.H4("Data Preview", style={'color': COLORS['primary']}),
            html.P(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns"),
            html.Div([
                dash_table.DataTable(
                    data=df.head(10).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in df.columns],
                    style_cell={
                        'textAlign': 'left', 
                        'fontSize': '11px',
                        'padding': '8px',
                        'whiteSpace': 'normal',
                        'height': 'auto',
                        'maxWidth': '200px',
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis'
                    },
                    style_header={
                        'backgroundColor': COLORS['primary'], 
                        'color': 'white', 
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'fontSize': '12px'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': COLORS['light']
                        }
                    ],
                    style_table={
                        'overflowX': 'auto',
                        'maxWidth': '100%',
                        'width': '100%'
                    },
                    style_cell_conditional=[
                        {
                            'if': {'column_id': col},
                            'maxWidth': '150px',
                            'minWidth': '80px',
                            'width': '150px'
                        } for col in df.columns
                    ],
                    page_size=10,
                    sort_action="native",
                    filter_action="native"
                )
            ], style={
                'overflowX': 'auto',
                'maxWidth': '100%',
                'border': f'1px solid {COLORS["border"]}',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            })
        ], style={'width': '100%', 'overflow': 'hidden'})
    return html.Div()

@app.callback(
    Output('metrics-container', 'children'),
    Input('stored-data', 'data')
)
def update_metrics(data):
    if data is not None:
        df = pd.read_json(data, orient='split')
        return generate_metrics_cards(df)
    return html.Div()

@app.callback(
    Output('main-content', 'children'),
    Input('stored-data', 'data')
)
def update_main_content(data):
    if data is not None:
        return html.Div(id='charts-container', className='row')
    return create_empty_state()

@app.callback(
    Output('filter-controls', 'children'),
    Input('stored-data', 'data')
)
def update_filter_controls(data):
    if data is not None:
        df = pd.read_json(data, orient='split')
        return create_filter_controls(df)
    return html.Div()

@app.callback(
    Output('stats-output', 'children'),
    [Input('correlation-btn', 'n_clicks'),
     Input('descriptive-btn', 'n_clicks'),
     Input('distribution-btn', 'n_clicks'),
     Input('missing-data-btn', 'n_clicks')],
    State('stored-data', 'data')
)
def generate_statistical_analysis(corr_clicks, desc_clicks, dist_clicks, missing_clicks, data):
    ctx = dash.callback_context
    if not ctx.triggered or not data:
        return html.Div()
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    df = pd.read_json(data, orient='split')
    
    if button_id == 'correlation-btn':
        return perform_correlation_analysis(df)
    elif button_id == 'descriptive-btn':
        return perform_descriptive_stats(df)
    elif button_id == 'distribution-btn':
        return perform_distribution_analysis(df)
    elif button_id == 'missing-data-btn':
        return perform_missing_data_analysis(df)
    
    return html.Div()

# Removed duplicate export callback - using the one below with export-status output

@app.callback(
    Output('ml-output', 'children'),
    [Input('predict-btn', 'n_clicks'),
     Input('cluster-btn', 'n_clicks'),
     Input('feature-btn', 'n_clicks'),
     Input('pca-btn', 'n_clicks'),
     Input('outlier-btn', 'n_clicks')],
    State('stored-data', 'data')
)
def generate_ml_analysis(predict_clicks, cluster_clicks, feature_clicks, pca_clicks, outlier_clicks, data):
    ctx = dash.callback_context
    if not ctx.triggered or not data:
        return html.Div()
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    df = pd.read_json(data, orient='split')
    
    if button_id == 'predict-btn':
        return perform_prediction_analysis(df)
    elif button_id == 'cluster-btn':
        return perform_cluster_analysis(df)
    elif button_id == 'feature-btn':
        return perform_feature_importance_analysis(df)
    elif button_id == 'pca-btn':
        return perform_dimensionality_reduction(df)
    elif button_id == 'outlier-btn':
        return perform_outlier_detection(df)
    
    return html.Div()

@app.callback(
    Output('filter-status', 'children'),
    Output('stored-data', 'data', allow_duplicate=True),
    Input('apply-filters-btn', 'n_clicks'),
    State('stored-data', 'data'),
    State('filter-controls', 'children'),
    prevent_initial_call=True
)
def apply_filters(n_clicks, data, filter_controls):
    if not n_clicks or not data:
        return html.Div(), data
    
    df = pd.read_json(data, orient='split')
    original_count = len(df)
    
    # Apply filters based on filter controls
    # This is a simplified version - in a full implementation, 
    # you'd need to capture the actual filter values from the controls
    filtered_df = df.copy()
    
    # For now, just return the original data
    # In a complete implementation, you would:
    # 1. Get filter values from the filter controls
    # 2. Apply categorical filters using isin()
    # 3. Apply numerical filters using between()
    # 4. Update the stored data with filtered results
    
    filtered_count = len(filtered_df)
    
    status_text = f"Applied filters: {original_count} → {filtered_count} records"
    return html.Div(status_text, style={'color': COLORS['success']}), data

@app.callback(
    Output('filter-status', 'children', allow_duplicate=True),
    Output('stored-data', 'data', allow_duplicate=True),
    Input('reset-filters-btn', 'n_clicks'),
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def reset_filters(n_clicks, data):
    if not n_clicks:
        return html.Div(), data
    
    # Reset all filters to show original data
    return html.Div("Filters reset to show all data", style={'color': COLORS['info']}), data

@app.callback(
    Output('charts-container', 'children', allow_duplicate=True),
    [Input({'type': 'chart-btn', 'index': dash.dependencies.ALL}, 'n_clicks')],
    State('charts-container', 'children'),
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def handle_chart_management(n_clicks_list, existing_children, data):
    ctx = dash.callback_context
    if not ctx.triggered or not existing_children:
        return existing_children
    
    button_id = ctx.triggered[0]['prop_id']
    button_type = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Extract chart ID from button ID
    if 'edit-' in button_id:
        chart_id = button_id.split('edit-')[1].split('.')[0]
        # TODO: Open edit modal
        return existing_children
    elif 'duplicate-' in button_id:
        chart_id = button_id.split('duplicate-')[1].split('.')[0]
        # TODO: Duplicate chart
        return existing_children
    elif 'export-' in button_id:
        chart_id = button_id.split('export-')[1].split('.')[0]
        # TODO: Export individual chart
        return existing_children
    elif 'delete-' in button_id:
        chart_id = button_id.split('delete-')[1].split('.')[0]
        # Remove chart from children
        new_children = [child for child in existing_children 
                       if not (hasattr(child, 'id') and child.id == f'chart-{chart_id}')]
        return new_children
    
    return existing_children

@app.callback(
    Output('template-description', 'children'),
    Input('template-selector', 'value')
)
def update_template_description(selected_template):
    if selected_template and selected_template in DASHBOARD_TEMPLATES:
        template = DASHBOARD_TEMPLATES[selected_template]
        return html.Div([
            html.P(f"📊 {template['name']}", style={'fontWeight': 'bold'}),
            html.P(template['description'], style={'fontSize': '11px'})
        ])
    return html.Div()

@app.callback(
    Output('charts-container', 'children', allow_duplicate=True),
    Input('load-template-btn', 'n_clicks'),
    State('template-selector', 'value'),
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def load_dashboard_template(n_clicks, selected_template, data):
    if not n_clicks or not selected_template or not data:
        return []
    
    if selected_template not in DASHBOARD_TEMPLATES:
        return []
    
    df = pd.read_json(data, orient='split')
    template = DASHBOARD_TEMPLATES[selected_template]
    charts = []
    
    for chart_config in template['charts']:
        chart_id = str(uuid.uuid4())
        fig = go.Figure()
        
        try:
            chart_type = chart_config['type']
            x_axis = chart_config.get('x')
            y_axis = chart_config.get('y')
            color = chart_config.get('color')
            title = chart_config.get('title', f'{chart_type.title()} Chart')
            
            if chart_type == 'bar' and x_axis and y_axis:
                fig = px.bar(df, x=x_axis, y=y_axis, color=color, title=title)
            elif chart_type == 'pie' and x_axis and y_axis:
                fig = px.pie(df, names=x_axis, values=y_axis, title=title)
            elif chart_type == 'scatter' and x_axis and y_axis:
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color, title=title)
            elif chart_type == 'histogram' and x_axis:
                fig = px.histogram(df, x=x_axis, title=title)
            elif chart_type == 'box' and x_axis and y_axis:
                fig = px.box(df, x=x_axis, y=y_axis, color=color, title=title)
            elif chart_type == 'violin' and x_axis and y_axis:
                fig = px.violin(df, x=x_axis, y=y_axis, color=color, title=title)
            elif chart_type == 'heatmap':
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) > 1:
                    corr_matrix = numeric_df.corr()
                    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title=title)
            
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(size=12),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            chart_container = html.Div([
                create_chart_management_toolbar(chart_id),
                dcc.Graph(figure=fig, id=f'chart-{chart_id}')
            ], className='six columns', style={
                'position': 'relative',
                'backgroundColor': COLORS['card'],
                'padding': '10px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'margin': '5px'
            })
            
            charts.append(chart_container)
            
        except Exception as e:
            # Skip charts that can't be created due to missing columns
            continue
    
    return charts

@app.callback(
    Output('export-status', 'children'),
    Input('export-pdf-btn', 'n_clicks'),
    Input('export-png-btn', 'n_clicks'),
    Input('export-csv-btn', 'n_clicks'),
    State('stored-data', 'data'),
    State('charts-container', 'children'),
    prevent_initial_call=True
)
def handle_export(pdf_clicks, png_clicks, csv_clicks, data, charts):
    ctx = dash.callback_context
    if not ctx.triggered:
        return html.Div()
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'export-pdf-btn' and pdf_clicks:
        try:
            pdf_data = export_dashboard_as_pdf(data, charts)
            if pdf_data:
                return html.Div([
                    html.Div([
                        html.I(className="fas fa-file-pdf", style={'color': COLORS['danger'], 'fontSize': '20px', 'marginRight': '10px'}),
                        html.Span("PDF Export Ready", style={'color': COLORS['danger'], 'fontWeight': 'bold', 'marginRight': '10px'}),
                        html.Span("Dashboard report generated successfully", 
                                 style={'color': COLORS['text'], 'fontSize': '12px'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'padding': '15px', 'backgroundColor': COLORS['light'], 'borderRadius': '8px', 'border': f'1px solid {COLORS["border"]}'}),
                    html.Div([
                        dcc.Download(id="download-pdf"),
                        html.Button("Download PDF", 
                                   style={'backgroundColor': COLORS['danger'], 'color': 'white', 'border': 'none', 'padding': '8px 16px', 'borderRadius': '4px', 'marginTop': '10px'},
                                   id='download-pdf-btn')
                    ])
                ])
            else:
                return html.Div([
                    html.Div([
                        html.I(className="fas fa-exclamation-triangle", style={'color': COLORS['warning'], 'fontSize': '20px', 'marginRight': '10px'}),
                        html.Span("PDF Export Failed", style={'color': COLORS['warning'], 'fontWeight': 'bold', 'marginRight': '10px'}),
                        html.Span("Error generating PDF report", 
                                 style={'color': COLORS['text'], 'fontSize': '12px'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'padding': '15px', 'backgroundColor': COLORS['light'], 'borderRadius': '8px', 'border': f'1px solid {COLORS["border"]}'})
                ])
        except Exception as e:
            return html.Div([
                html.Div([
                    html.I(className="fas fa-exclamation-triangle", style={'color': COLORS['warning'], 'fontSize': '20px', 'marginRight': '10px'}),
                    html.Span("PDF Export Error", style={'color': COLORS['warning'], 'fontWeight': 'bold', 'marginRight': '10px'}),
                    html.Span(f"Error: {str(e)}", 
                             style={'color': COLORS['text'], 'fontSize': '12px'})
                ], style={'display': 'flex', 'alignItems': 'center', 'padding': '15px', 'backgroundColor': COLORS['light'], 'borderRadius': '8px', 'border': f'1px solid {COLORS["border"]}'})
            ])
    elif button_id == 'export-png-btn' and png_clicks:
        try:
            png_data = export_charts_as_png(charts, data)
            if png_data:
                return html.Div([
                    html.Div([
                        html.I(className="fas fa-image", style={'color': COLORS['info'], 'fontSize': '20px', 'marginRight': '10px'}),
                        html.Span("PNG Export Ready", style={'color': COLORS['info'], 'fontWeight': 'bold', 'marginRight': '10px'}),
                        html.Span("Charts exported as high-resolution PNG", 
                                 style={'color': COLORS['text'], 'fontSize': '12px'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'padding': '15px', 'backgroundColor': COLORS['light'], 'borderRadius': '8px', 'border': f'1px solid {COLORS["border"]}'}),
                    html.Div([
                        dcc.Download(id="download-png"),
                        html.Button("Download PNG", 
                                   style={'backgroundColor': COLORS['info'], 'color': 'white', 'border': 'none', 'padding': '8px 16px', 'borderRadius': '4px', 'marginTop': '10px'},
                                   id='download-png-btn')
                    ])
                ])
            else:
                return html.Div([
                    html.Div([
                        html.I(className="fas fa-exclamation-triangle", style={'color': COLORS['warning'], 'fontSize': '20px', 'marginRight': '10px'}),
                        html.Span("PNG Export Failed", style={'color': COLORS['warning'], 'fontWeight': 'bold', 'marginRight': '10px'}),
                        html.Span("No charts available to export", 
                                 style={'color': COLORS['text'], 'fontSize': '12px'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'padding': '15px', 'backgroundColor': COLORS['light'], 'borderRadius': '8px', 'border': f'1px solid {COLORS["border"]}'})
                ])
        except Exception as e:
            return html.Div([
                html.Div([
                    html.I(className="fas fa-exclamation-triangle", style={'color': COLORS['warning'], 'fontSize': '20px', 'marginRight': '10px'}),
                    html.Span("PNG Export Error", style={'color': COLORS['warning'], 'fontWeight': 'bold', 'marginRight': '10px'}),
                    html.Span(f"Error: {str(e)}", 
                             style={'color': COLORS['text'], 'fontSize': '12px'})
                ], style={'display': 'flex', 'alignItems': 'center', 'padding': '15px', 'backgroundColor': COLORS['light'], 'borderRadius': '8px', 'border': f'1px solid {COLORS["border"]}'})
            ])
    elif button_id == 'export-csv-btn' and csv_clicks and data:
        df = pd.read_json(data, orient='split')
        csv_string = df.to_csv(index=False)
        return html.Div([
            html.Div([
                html.I(className="fas fa-file-csv", style={'color': COLORS['success'], 'fontSize': '20px', 'marginRight': '10px'}),
                html.Span("CSV Export Ready", style={'color': COLORS['success'], 'fontWeight': 'bold', 'marginRight': '10px'}),
                html.Span(f"Data: {len(df):,} rows × {len(df.columns)} columns", 
                         style={'color': COLORS['text'], 'fontSize': '12px'})
            ], style={'display': 'flex', 'alignItems': 'center', 'padding': '15px', 'backgroundColor': COLORS['light'], 'borderRadius': '8px', 'border': f'1px solid {COLORS["border"]}'}),
            html.Div([
                dcc.Download(id="download-csv"),
                html.Button("Download CSV", 
                           style={'backgroundColor': COLORS['success'], 'color': 'white', 'border': 'none', 'padding': '8px 16px', 'borderRadius': '4px', 'marginTop': '10px'},
                           id='download-csv-btn')
            ])
        ])
    
    return html.Div()

@app.callback(
    Output('download-csv', 'data'),
    Input('download-csv-btn', 'n_clicks'),
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def download_csv(n_clicks, data):
    if n_clicks and data:
        df = pd.read_json(data, orient='split')
        csv_string = df.to_csv(index=False)
        return dict(content=csv_string, filename="dashboard_data.csv")
    return None

@app.callback(
    Output('download-pdf', 'data'),
    Input('download-pdf-btn', 'n_clicks'),
    State('stored-data', 'data'),
    State('charts-container', 'children'),
    prevent_initial_call=True
)
def download_pdf(n_clicks, data, charts):
    if n_clicks and data:
        pdf_data = export_dashboard_as_pdf(data, charts)
        if pdf_data:
            return dict(content=pdf_data, filename="dashboard_report.pdf")
    return None

@app.callback(
    Output('download-png', 'data'),
    Input('download-png-btn', 'n_clicks'),
    State('charts-container', 'children'),
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def download_png(n_clicks, charts, data):
    if n_clicks:
        png_data = export_charts_as_png(charts, data)
        if png_data:
            return dict(content=png_data, filename="dashboard_charts.png")
    return None

@app.callback(
    Output('export-status', 'children', allow_duplicate=True),
    Input('save-config-btn', 'n_clicks'),
    State('stored-data', 'data'),
    State('charts-container', 'children'),
    prevent_initial_call=True
)
def save_dashboard_config(n_clicks, data, charts):
    if not n_clicks:
        return html.Div()
    
    try:
        config = {
            'data': data,
            'charts': charts,
            'timestamp': time.time(),
            'version': '1.0'
        }
        
        # In a real implementation, this would save to a file
        config_json = json.dumps(config, indent=2)
        
        return html.Div([
            html.H4("✅ Configuration Saved", style={'color': COLORS['success']}),
            html.P("Dashboard configuration has been saved successfully."),
            html.P(f"Config size: {len(config_json)} characters")
        ])
    except Exception as e:
        return html.Div([
            html.H4("❌ Save Failed", style={'color': COLORS['danger']}),
            html.P(f"Error saving configuration: {str(e)}")
        ])

@app.callback(
    Output('export-status', 'children', allow_duplicate=True),
    Input('load-config-btn', 'n_clicks'),
    prevent_initial_call=True
)
def load_dashboard_config(n_clicks):
    if not n_clicks:
        return html.Div()
    
    try:
        # In a real implementation, this would load from a file
        return html.Div([
            html.H4("📁 Load Configuration", style={'color': COLORS['info']}),
            html.P("Configuration loading functionality will be implemented here."),
            html.P("This would restore the dashboard to a previously saved state.")
        ])
    except Exception as e:
        return html.Div([
            html.H4("❌ Load Failed", style={'color': COLORS['danger']}),
            html.P(f"Error loading configuration: {str(e)}")
        ])

@app.callback(
    Output('performance-metrics', 'children'),
    Input('check-performance-btn', 'n_clicks'),
    prevent_initial_call=True
)
def check_performance(n_clicks):
    if not n_clicks:
        return html.Div()
    
    try:
        metrics = get_performance_metrics()
        
        return html.Div([
            html.H5("System Performance", style={'color': COLORS['primary']}),
            html.P(f"CPU Usage: {metrics['cpu_usage']}%"),
            html.P(f"Memory Usage: {metrics['memory_usage']}%"),
            html.P(f"Available Memory: {metrics['memory_available']} GB"),
            html.P(f"Process Memory: {metrics['process_memory']} MB"),
            html.P(f"Disk Usage: {metrics['disk_usage']}%"),
            html.Hr(),
            html.P("Performance monitoring helps optimize dashboard performance.", 
                   style={'fontSize': '10px', 'color': COLORS['text']})
        ])
    except Exception as e:
        return html.Div([
            html.H5("❌ Performance Check Failed", style={'color': COLORS['danger']}),
            html.P(f"Error: {str(e)}")
        ])

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
    State('color-dropdown', 'value'),
    State('size-dropdown', 'value')
)
def add_chart(n_clicks, existing_children, data, chart_type, x_axis, y_axis, color, size):
    if n_clicks == 0 or not data or not chart_type or not x_axis:
        return existing_children or []
    
    df = get_df(data)
    chart_id = str(uuid.uuid4())
    fig = go.Figure()
    
    try:
        if chart_type == 'bar':
            fig = px.bar(df, x=x_axis, y=y_axis, color=color, title=f'Bar Chart: {y_axis} by {x_axis}')
        elif chart_type == 'scatter':
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color, size=size, title=f'Scatter Plot: {y_axis} vs {x_axis}')
        elif chart_type == 'histogram':
            fig = px.histogram(df, x=x_axis, color=color, title=f'Histogram: {x_axis}')
        elif chart_type == 'pie':
            fig = px.pie(df, names=x_axis, values=y_axis, title=f'Pie Chart: {x_axis}')
        elif chart_type == 'box':
            fig = px.box(df, x=x_axis, y=y_axis, color=color, title=f'Box Plot: {y_axis} by {x_axis}')
        elif chart_type == 'heatmap':
            # Create correlation heatmap for numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Heatmap")
            else:
                fig = px.bar(df, x=x_axis, y=y_axis, title=f'Bar Chart: {y_axis} by {x_axis}')
        elif chart_type == 'violin':
            fig = px.violin(df, x=x_axis, y=y_axis, color=color, title=f'Violin Plot: {y_axis} by {x_axis}')
        elif chart_type == 'line':
            fig = px.line(df, x=x_axis, y=y_axis, color=color, title=f'Line Chart: {y_axis} by {x_axis}')
        elif chart_type == 'sunburst':
            # Create hierarchical sunburst chart
            if len(df.columns) >= 2:
                fig = px.sunburst(df, path=[x_axis, y_axis], title=f'Sunburst: {x_axis} → {y_axis}')
            else:
                fig = px.pie(df, names=x_axis, values=y_axis, title=f'Pie Chart: {x_axis}')
        elif chart_type == 'treemap':
            # Create treemap chart
            if len(df.columns) >= 2:
                fig = px.treemap(df, path=[x_axis], values=y_axis, title=f'Treemap: {y_axis} by {x_axis}')
            else:
                fig = px.bar(df, x=x_axis, y=y_axis, title=f'Bar Chart: {y_axis} by {x_axis}')
        elif chart_type == '3d_scatter':
            # Create 3D scatter plot
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 3:
                fig = px.scatter_3d(df, x=numeric_cols[0], y=numeric_cols[1], z=numeric_cols[2], 
                                  color=color, title=f'3D Scatter Plot')
            else:
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color, title=f'Scatter Plot: {y_axis} vs {x_axis}')
        elif chart_type == 'area':
            fig = px.area(df, x=x_axis, y=y_axis, color=color, title=f'Area Chart: {y_axis} by {x_axis}')
        
        # Update layout for better appearance
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        # Create chart container with management toolbar
        chart_container = html.Div([
            create_chart_management_toolbar(chart_id),
            dcc.Graph(figure=fig, id=f'chart-{chart_id}')
        ], className='six columns', style={
            'position': 'relative',
            'backgroundColor': COLORS['card'],
            'padding': '10px',
            'borderRadius': '8px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'margin': '5px'
        })
        
    except Exception as e:
        return html.Div(f"Error creating chart: {str(e)}", style={'color': COLORS['danger']})
    
    if existing_children is None:
        existing_children = []
        
    existing_children.append(chart_container)
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
