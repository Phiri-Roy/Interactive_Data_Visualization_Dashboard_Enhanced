"""
Interactive Data Visualization Dashboard Using Python

Project Description:
This project demonstrates comprehensive data analysis and visualization techniques using Python.
We analyze the famous Titanic dataset to uncover insights about passenger demographics, 
survival patterns, and relationships between various factors.

Objectives:
- Data Analysis: Perform thorough data preprocessing and exploratory data analysis
- Static Visualizations: Create informative charts using Matplotlib and Seaborn
- Interactive Visualizations: Build dynamic charts using Plotly for enhanced user experience
- Comprehensive Coverage: Include at least 5 different chart types (bar, line, pie, heatmap, scatter)
- Real-world Application: Use authentic dataset with proper data cleaning and preprocessing

Dataset:
We use the Titanic dataset, which contains information about passengers aboard the RMS Titanic.
This dataset includes demographics, ticket information, and survival outcomes.

Visualization Types:
1. Bar Chart - Passenger distribution by class (Matplotlib)
2. Interactive Line Chart - Average fare trends across age groups (Plotly)
3. Pie Chart - Survival rate distribution (Matplotlib)
4. Heatmap - Correlation matrix of numerical features (Seaborn)
5. Interactive Scatter Plot - Age vs Fare relationship with survival status (Plotly)

Author: Python Data Analyst
Date: 2024
"""

# ============================================================================
# 1. LIBRARY IMPORTS AND SETUP
# ============================================================================

# Import required libraries with error handling
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import warnings
    
    # Configure display settings
    plt.style.use('default')
    sns.set_palette("husl")
    warnings.filterwarnings('ignore')
    
    # Set figure parameters for better visualization
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    
    print("[SUCCESS] All libraries imported successfully!")
    print(f"[INFO] Pandas version: {pd.__version__}")
    print(f"[INFO] Matplotlib version: {plt.matplotlib.__version__}")
    print(f"[INFO] Seaborn version: {sns.__version__}")
    print("=" * 60)
    
except ImportError as e:
    print(f"[ERROR] Error importing libraries: {e}")
    print("Please install missing packages using: pip install pandas matplotlib seaborn plotly")
    exit(1)

# ============================================================================
# 2. DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

def load_and_explore_data():
    """Load the Titanic dataset and perform initial exploration"""
    try:
        # Load dataset using seaborn's built-in Titanic dataset
        df = sns.load_dataset('titanic')
        
        print("[SUCCESS] Dataset loaded successfully!")
        print(f"[DATA] Dataset shape: {df.shape}")
        print(f"[INFO] Columns: {list(df.columns)}")
        
        # Display first few rows
        print("\n[ANALYSIS] First 5 rows of the dataset:")
        print(df.head())
        
        # Basic dataset information
        print("\n[CHART] Dataset Info:")
        print(df.info())
        
        # Display basic statistics
        print("\n[INFO] Basic Statistics:")
        print(df.describe())
        
        return df
        
    except Exception as e:
        print(f"[ERROR] Error loading dataset: {e}")
        print("Please ensure you have an internet connection to download the dataset.")
        return None

# ============================================================================
# 3. DATA PREPROCESSING AND CLEANING
# ============================================================================

def preprocess_data(df):
    """Clean and preprocess the dataset"""
    try:
        print("\n" + "="*60)
        print("[TOOLS] DATA PREPROCESSING AND CLEANING")
        print("="*60)
        
        # Check for missing values
        print("[ANALYSIS] Missing values analysis:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])
        
        # Create a copy for preprocessing
        df_clean = df.copy()
        
        # Handle missing values
        # Fill missing age values with median
        if 'age' in df_clean.columns:
            median_age = df_clean['age'].median()
            missing_age_count = df['age'].isnull().sum()
            df_clean['age'].fillna(median_age, inplace=True)
            print(f"[SUCCESS] Filled {missing_age_count} missing age values with median: {median_age:.1f}")
        
        # Fill missing fare values with median
        if 'fare' in df_clean.columns:
            median_fare = df_clean['fare'].median()
            missing_fare_count = df['fare'].isnull().sum()
            df_clean['fare'].fillna(median_fare, inplace=True)
            print(f"[SUCCESS] Filled {missing_fare_count} missing fare values with median: {median_fare:.2f}")
        
        # Fill missing embarked values with mode
        if 'embarked' in df_clean.columns:
            mode_embarked = df_clean['embarked'].mode()[0]
            missing_embarked_count = df['embarked'].isnull().sum()
            df_clean['embarked'].fillna(mode_embarked, inplace=True)
            print(f"[SUCCESS] Filled {missing_embarked_count} missing embarked values with mode: {mode_embarked}")
        
        # Create age groups for better analysis
        df_clean['age_group'] = pd.cut(df_clean['age'], 
                                       bins=[0, 12, 18, 35, 60, 100], 
                                       labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
        
        # Create fare categories
        df_clean['fare_category'] = pd.cut(df_clean['fare'], 
                                           bins=[0, 10, 30, 100, 1000], 
                                           labels=['Low', 'Medium', 'High', 'Premium'])
        
        # Create family size feature
        df_clean['family_size'] = df_clean['sibsp'] + df_clean['parch'] + 1
        df_clean['family_category'] = pd.cut(df_clean['family_size'], 
                                            bins=[0, 1, 4, 20], 
                                            labels=['Alone', 'Small Family', 'Large Family'])
        
        print(f"\n[SUCCESS] Data preprocessing completed successfully!")
        print(f"[INFO] Clean dataset shape: {df_clean.shape}")
        print(f"[ANALYSIS] Remaining missing values: {df_clean.isnull().sum().sum()}")
        
        return df_clean
        
    except Exception as e:
        print(f"[ERROR] Error during preprocessing: {e}")
        return df

# ============================================================================
# 4. VISUALIZATION 1: BAR CHART - PASSENGER DISTRIBUTION BY CLASS (MATPLOTLIB)
# ============================================================================

def create_bar_chart(df_clean):
    """Create bar chart showing passenger distribution by class"""
    try:
        print("\n" + "="*60)
        print("[INFO] VISUALIZATION 1: BAR CHART - PASSENGER DISTRIBUTION BY CLASS")
        print("="*60)
        
        # Calculate passenger counts by class
        class_counts = df_clean['class'].value_counts().sort_index()
        
        # Create the bar chart
        plt.figure(figsize=(12, 8))
        bars = plt.bar(class_counts.index, class_counts.values, 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1'], 
                       edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Customize the chart
        plt.title('Passenger Distribution by Class', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Passenger Class', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Passengers', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Add grid for better readability
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Display the chart
        plt.show()
        
        print("[INFO] Bar Chart Analysis:")
        for class_name, count in class_counts.items():
            percentage = (count / len(df_clean)) * 100
            print(f"   {class_name}: {count} passengers ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"[ERROR] Error creating bar chart: {e}")

# ============================================================================
# 5. VISUALIZATION 2: INTERACTIVE LINE CHART - AVERAGE FARE BY AGE GROUP (PLOTLY)
# ============================================================================

def create_interactive_line_chart(df_clean):
    """Create interactive line chart showing average fare across age groups"""
    try:
        print("\n" + "="*60)
        print("[CHART] VISUALIZATION 2: INTERACTIVE LINE CHART - AVERAGE FARE BY AGE GROUP")
        print("="*60)
        
        # Create age bins for analysis
        df_clean['age_bin'] = pd.cut(df_clean['age'], bins=range(0, 90, 10), right=False)
        
        # Calculate average fare by age group
        fare_by_age = df_clean.groupby('age_bin').agg({
            'fare': ['mean', 'count', 'std']
        }).round(2)
        
        fare_by_age.columns = ['avg_fare', 'passenger_count', 'fare_std']
        fare_by_age = fare_by_age.reset_index()
        
        # Convert age_bin to string for better display
        fare_by_age['age_range'] = fare_by_age['age_bin'].astype(str)
        
        # Create interactive line chart with Plotly
        fig = px.line(fare_by_age, x='age_range', y='avg_fare',
                      title='Average Fare Across Age Groups (Interactive)',
                      labels={'age_range': 'Age Range', 'avg_fare': 'Average Fare ($)'},
                      markers=True, line_shape='spline')
        
        # Customize the chart
        fig.update_traces(line=dict(color='#FF6B6B', width=4),
                          marker=dict(size=10, color='#4ECDC4', 
                                      line=dict(width=2, color='#FF6B6B')))
        
        # Add hover information
        fig.update_traces(hovertemplate=
                          '<b>Age Range:</b> %{x}<br>' +
                          '<b>Average Fare:</b> $%{y:.2f}<br>' +
                          '<extra></extra>')
        
        # Update layout
        fig.update_layout(
            title=dict(font=dict(size=20, color='#2C3E50'), x=0.5),
            xaxis=dict(title=dict(font=dict(size=14, color='#2C3E50')),
                       tickangle=45),
            yaxis=dict(title=dict(font=dict(size=14, color='#2C3E50'))),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=600
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        # Display the interactive chart
        fig.show()
        
        print("[CHART] Interactive Line Chart Analysis:")
        print("   Hover over the points to see detailed information!")
        print(f"   Highest average fare: ${fare_by_age['avg_fare'].max():.2f}")
        print(f"   Lowest average fare: ${fare_by_age['avg_fare'].min():.2f}")
        
    except Exception as e:
        print(f"[ERROR] Error creating interactive line chart: {e}")

# ============================================================================
# 6. VISUALIZATION 3: PIE CHART - SURVIVAL DISTRIBUTION (MATPLOTLIB)
# ============================================================================

def create_pie_chart(df_clean):
    """Create pie chart showing survival distribution"""
    try:
        print("\n" + "="*60)
        print("🥧 VISUALIZATION 3: PIE CHART - SURVIVAL DISTRIBUTION")
        print("="*60)
        
        # Calculate survival counts
        survival_counts = df_clean['survived'].value_counts()
        labels = ['Did Not Survive', 'Survived']
        colors = ['#FF6B6B', '#4ECDC4']
        explode = (0.05, 0.05)  # Slightly separate the slices
        
        # Create the pie chart
        plt.figure(figsize=(12, 10))
        wedges, texts, autotexts = plt.pie(survival_counts.values, 
                                           labels=labels,
                                           colors=colors,
                                           autopct='%1.1f%%',
                                           startangle=90,
                                           explode=explode,
                                           shadow=True,
                                           textprops={'fontsize': 14, 'fontweight': 'bold'})
        
        # Customize the chart
        plt.title('Titanic Survival Distribution', fontsize=18, fontweight='bold', pad=20)
        
        # Make percentage text more visible
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(16)
            autotext.set_fontweight('bold')
        
        # Add a legend with counts
        legend_labels = [f'{label}: {count} passengers' for label, count in zip(labels, survival_counts.values)]
        plt.legend(wedges, legend_labels, title="Survival Status", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.tight_layout()
        plt.show()
        
        # Print survival statistics
        total_passengers = len(df_clean)
        survived = survival_counts[1]
        not_survived = survival_counts[0]
        survival_rate = (survived / total_passengers) * 100
        
        print("[SHIP] Survival Analysis:")
        print(f"   Total Passengers: {total_passengers}")
        print(f"   Survived: {survived} ({survival_rate:.1f}%)")
        print(f"   Did Not Survive: {not_survived} ({100-survival_rate:.1f}%)")
        
    except Exception as e:
        print(f"[ERROR] Error creating pie chart: {e}")

# ============================================================================
# 7. VISUALIZATION 4: HEATMAP - CORRELATION MATRIX (SEABORN)
# ============================================================================

def create_heatmap(df_clean):
    """Create heatmap showing correlation matrix of numerical features"""
    try:
        print("\n" + "="*60)
        print("🔥 VISUALIZATION 4: HEATMAP - CORRELATION MATRIX")
        print("="*60)
        
        # Select numerical columns for correlation analysis
        numerical_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
        
        # Calculate correlation matrix
        correlation_matrix = df_clean[numerical_cols].corr()
        
        # Create the heatmap
        plt.figure(figsize=(14, 10))
        
        # Create mask for upper triangle to show only lower triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Generate heatmap
        sns.heatmap(correlation_matrix, 
                    mask=mask,
                    annot=True, 
                    fmt='.3f', 
                    cmap='RdYlBu_r',
                    center=0,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": .8},
                    annot_kws={'fontsize': 12, 'fontweight': 'bold'})
        
        # Customize the chart
        plt.title('Correlation Heatmap of Numerical Features', 
                  fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Features', fontsize=14, fontweight='bold')
        plt.ylabel('Features', fontsize=14, fontweight='bold')
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.show()
        
        # Analyze strongest correlations
        print("[ANALYSIS] Correlation Analysis:")
        
        # Find strongest positive and negative correlations (excluding self-correlations)
        corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_pairs.append((
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    correlation_matrix.iloc[i, j]
                ))
        
        # Sort by absolute correlation value
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        print("   Top 5 strongest correlations:")
        for i, (var1, var2, corr) in enumerate(corr_pairs[:5]):
            print(f"   {i+1}. {var1} ↔ {var2}: {corr:.3f}")
        
    except Exception as e:
        print(f"[ERROR] Error creating heatmap: {e}")

# ============================================================================
# 8. VISUALIZATION 5: INTERACTIVE SCATTER PLOT - AGE VS FARE (PLOTLY)
# ============================================================================

def create_interactive_scatter_plot(df_clean):
    """Create interactive scatter plot showing Age vs Fare with survival status"""
    try:
        print("\n" + "="*60)
        print("🎯 VISUALIZATION 5: INTERACTIVE SCATTER PLOT - AGE VS FARE")
        print("="*60)
        
        # Prepare data for scatter plot
        df_scatter = df_clean.copy()
        df_scatter['survival_status'] = df_scatter['survived'].map({0: 'Did Not Survive', 1: 'Survived'})
        df_scatter['gender'] = df_scatter['sex'].str.title()
        
        # Create interactive scatter plot
        fig = px.scatter(df_scatter, 
                         x='age', 
                         y='fare',
                         color='survival_status',
                         size='pclass',
                         hover_data=['gender', 'class', 'embarked'],
                         title='Interactive Scatter Plot: Age vs Fare (Colored by Survival Status)',
                         labels={
                             'age': 'Age (years)',
                             'fare': 'Fare ($)',
                             'survival_status': 'Survival Status',
                             'pclass': 'Passenger Class'
                         },
                         color_discrete_map={
                             'Survived': '#4ECDC4',
                             'Did Not Survive': '#FF6B6B'
                         })
        
        # Customize the scatter plot
        fig.update_traces(marker=dict(line=dict(width=1, color='white'),
                                      opacity=0.7))
        
        # Update layout
        fig.update_layout(
            title=dict(font=dict(size=20, color='#2C3E50'), x=0.5),
            xaxis=dict(title=dict(font=dict(size=14, color='#2C3E50')),
                       showgrid=True, gridwidth=1, gridcolor='lightgray'),
            yaxis=dict(title=dict(font=dict(size=14, color='#2C3E50')),
                       showgrid=True, gridwidth=1, gridcolor='lightgray'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=700,
            legend=dict(title=dict(font=dict(size=12, color='#2C3E50')),
                        font=dict(size=11))
        )
        
        # Display the interactive scatter plot
        fig.show()
        
        # Statistical analysis
        print("[INFO] Scatter Plot Analysis:")
        
        # Survival rates by age groups
        age_survival = df_clean.groupby('age_group')['survived'].agg(['count', 'sum', 'mean']).round(3)
        age_survival.columns = ['Total', 'Survived', 'Survival_Rate']
        
        print("   Survival rates by age group:")
        for idx, row in age_survival.iterrows():
            print(f"   {idx}: {row['Survival_Rate']:.1%} ({int(row['Survived'])}/{int(row['Total'])} passengers)")
        
        # Fare statistics by survival
        fare_stats = df_clean.groupby('survived')['fare'].agg(['mean', 'median', 'std']).round(2)
        print("\n   Average fare by survival status:")
        print(f"   Did Not Survive: ${fare_stats.loc[0, 'mean']:.2f} (median: ${fare_stats.loc[0, 'median']:.2f})")
        print(f"   Survived: ${fare_stats.loc[1, 'mean']:.2f} (median: ${fare_stats.loc[1, 'median']:.2f})")
        
    except Exception as e:
        print(f"[ERROR] Error creating interactive scatter plot: {e}")

# ============================================================================
# 9. COMPREHENSIVE SURVIVAL ANALYSIS
# ============================================================================

def comprehensive_survival_analysis(df_clean):
    """Perform comprehensive survival analysis"""
    try:
        print("\n" + "="*60)
        print("[ANALYSIS] COMPREHENSIVE SURVIVAL ANALYSIS")
        print("="*60)
        
        # 1. Survival by Gender
        gender_survival = df_clean.groupby('sex')['survived'].agg(['count', 'sum', 'mean'])
        gender_survival.columns = ['Total', 'Survived', 'Survival_Rate']
        
        print("\n👥 Survival by Gender:")
        for gender, row in gender_survival.iterrows():
            print(f"   {gender.title()}: {row['Survival_Rate']:.1%} ({int(row['Survived'])}/{int(row['Total'])} passengers)")
        
        # 2. Survival by Class
        class_survival = df_clean.groupby('class')['survived'].agg(['count', 'sum', 'mean'])
        class_survival.columns = ['Total', 'Survived', 'Survival_Rate']
        
        print("\n🎫 Survival by Class:")
        for class_name, row in class_survival.iterrows():
            print(f"   {class_name}: {row['Survival_Rate']:.1%} ({int(row['Survived'])}/{int(row['Total'])} passengers)")
        
        # 3. Survival by Embarkation Port
        embark_survival = df_clean.groupby('embarked')['survived'].agg(['count', 'sum', 'mean'])
        embark_survival.columns = ['Total', 'Survived', 'Survival_Rate']
        
        print("\n[SHIP] Survival by Embarkation Port:")
        for port, row in embark_survival.iterrows():
            print(f"   {port}: {row['Survival_Rate']:.1%} ({int(row['Survived'])}/{int(row['Total'])} passengers)")
        
        # 4. Family Size Analysis
        family_survival = df_clean.groupby('family_category')['survived'].agg(['count', 'sum', 'mean'])
        family_survival.columns = ['Total', 'Survived', 'Survival_Rate']
        
        print("\n👨‍👩‍👧‍👦 Survival by Family Size:")
        for category, row in family_survival.iterrows():
            print(f"   {category}: {row['Survival_Rate']:.1%} ({int(row['Survived'])}/{int(row['Total'])} passengers)")
        
        # 5. Age Group Analysis
        print("\n👶👦👨👴 Survival by Age Group:")
        for idx, row in df_clean.groupby('age_group')['survived'].agg(['count', 'sum', 'mean']).iterrows():
            print(f"   {idx}: {row['mean']:.1%} ({int(row['sum'])}/{int(row['count'])} passengers)")
        
    except Exception as e:
        print(f"[ERROR] Error in comprehensive analysis: {e}")

# ============================================================================
# 10. MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Main function to execute the complete data visualization dashboard"""
    print("[SHIP]" * 20)
    print("INTERACTIVE DATA VISUALIZATION DASHBOARD")
    print("TITANIC DATASET ANALYSIS")
    print("[SHIP]" * 20)
    
    # Load and explore data
    df = load_and_explore_data()
    if df is None:
        return
    
    # Preprocess data
    df_clean = preprocess_data(df)
    
    # Create visualizations
    create_bar_chart(df_clean)
    create_interactive_line_chart(df_clean)
    create_pie_chart(df_clean)
    create_heatmap(df_clean)
    create_interactive_scatter_plot(df_clean)
    
    # Comprehensive analysis
    comprehensive_survival_analysis(df_clean)
    
    # Conclusion
    print("\n" + "="*60)
    print("[DATA] PROJECT CONCLUSION")
    print("="*60)
    print("""
🎯 KEY FINDINGS:
   • Women had significantly higher survival rates than men (74.2% vs 18.9%)
   • First-class passengers had the highest survival rate (62.9%)
   • Children had better survival chances than adults
   • Passengers who paid higher fares generally had better survival rates
   • Family size affected survival - small families had better chances than solo travelers

[INFO] VISUALIZATIONS CREATED:
   [SUCCESS] Bar Chart - Passenger distribution by class (Matplotlib)
   [SUCCESS] Interactive Line Chart - Average fare by age groups (Plotly)
   [SUCCESS] Pie Chart - Overall survival distribution (Matplotlib)
   [SUCCESS] Heatmap - Correlation matrix of numerical features (Seaborn)
   [SUCCESS] Interactive Scatter Plot - Age vs Fare with survival status (Plotly)

[TOOLS] TECHNICAL ACHIEVEMENTS:
   • Comprehensive data preprocessing and cleaning
   • Multiple visualization libraries integration
   • Interactive charts with hover functionality
   • Statistical analysis and insights generation
   • Error handling and robust code structure

🚀 FUTURE ENHANCEMENTS:
   • Add more interactive filters and controls
   • Implement machine learning models for survival prediction
   • Create a web-based dashboard using Dash or Streamlit
   • Include more advanced statistical analyses
   • Add data export functionality
    """)
    
    print("[SUCCESS] Dashboard execution completed successfully!")
    print("Thank you for using the Interactive Data Visualization Dashboard! 🎉")

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    main()
