import snowflake.connector
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from jinja2 import Template
import datetime
import re
import os

# Snowflake connection parameters
SNOWFLAKE_CONFIG = {
    'user': 'TES_USR_LACAK',
    'password': 'StrongPassword123',
    'account': 'hb01677.ap-southeast-3.aws',
    'warehouse': 'COMPUTE_WH',
    'database': 'VEHICLE_TRACKING',
    'schema': 'PUBLIC'
}

# Function to connect to Snowflake and fetch data
def fetch_data_from_snowflake():
    try:
        conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
        query = """
        select * from tes_db_raw.raw_lacak.tr_track
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error connecting to Snowflake: {e}")
        return None

# Function to convert duration (HH:MM:SS) to seconds
def duration_to_seconds(duration):
    if pd.isna(duration):
        return 0
    try:
        h, m, s = map(int, duration.split(':'))
        return h * 3600 + m * 60 + s
    except:
        return 0

# Function to extract mileage from DESCRIPTION
def extract_mileage(description):
    if pd.isna(description):
        return 0
    match = re.search(r'Mileage: (\d+\.?\d*)', str(description))
    return float(match.group(1)) if match else 0

# Preprocessing and feature engineering
def preprocess_data(df):
    # Handle missing values
    df = df.dropna(subset=['UNIT', 'TYPE', 'DATE_TIME_BEGIN', 'DATE_TIME_END'])
    
    # Parse dates
    df['DATE_TIME_BEGIN'] = pd.to_datetime(df['DATE_TIME_BEGIN'])
    df['DATE_TIME_END'] = pd.to_datetime(df['DATE_TIME_END'])
    
    # Extract temporal features
    df['HOUR_OF_DAY'] = df['DATE_TIME_BEGIN'].dt.hour
    df['DAY_OF_WEEK'] = df['DATE_TIME_BEGIN'].dt.dayofweek
    
    # Convert duration to seconds
    df['DURATION_SECONDS'] = df['DURATION'].apply(duration_to_seconds)
    
    # Extract mileage
    df['MILEAGE'] = df['DESCRIPTION'].apply(extract_mileage)
    
    # Encode categorical variables
    le_type = LabelEncoder()
    le_location = LabelEncoder()
    df['TYPE_ENCODED'] = le_type.fit_transform(df['TYPE'])
    df['INITIAL_LOCATION_ENCODED'] = le_location.fit_transform(df['INITIAL_LOCATION'])
    
    return df, le_type, le_location

# Apply DBSCAN clustering
def apply_dbscan(df):
    features = df[['DURATION_SECONDS', 'MILEAGE', 'HOUR_OF_DAY', 'DAY_OF_WEEK', 'TYPE_ENCODED', 'INITIAL_LOCATION_ENCODED']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(scaled_features)
    
    df['CLUSTER'] = clusters
    return df

# Generate HTML report with Recharts
def generate_html_report(df, le_type, le_location):
    # Aggregate data for visualizations
    type_counts = df['TYPE'].value_counts().to_dict()
    type_chart_data = [{'type': k, 'count': v} for k, v in type_counts.items()]
    
    location_counts = df['INITIAL_LOCATION'].value_counts().head(5).to_dict()
    location_chart_data = [{'location': k, 'count': v} for k, v in location_counts.items()]
    
    cluster_counts = df['CLUSTER'].value_counts().to_dict()
    cluster_chart_data = [{'cluster': str(k), 'count': v} for k, v in cluster_counts.items() if k != -1]  # Exclude noise
    
    # HTML template with Recharts
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vehicle Tracking Analysis Report</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/prop-types/15.8.1/prop-types.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/recharts/2.15.0/Recharts.min.js"></script>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body { font-family: Arial, sans-serif; }
            .chart-container { margin: 20px 0; }
        </style>
    </head>
    <body class="bg-gray-100 p-6">
        <div id="root"></div>
        <script>
            const typeChartData = {{ type_chart_data | tojson }};
            const locationChartData = {{ location_chart_data | tojson }};
            const clusterChartData = {{ cluster_chart_data | tojson }};

            const App = () => (
                <div className="max-w-4xl mx-auto bg-white p-6 rounded-lg shadow-md">
                    <h1 className="text-3xl font-bold mb-4">Vehicle Tracking Analysis Report</h1>
                    <p className="mb-4">
                        This report analyzes vehicle tracking data using DBSCAN clustering to identify patterns
                        in trips and parking events. Key insights include frequent locations, event types,
                        and cluster distributions.
                    </p>

                    <div className="chart-container">
                        <h2 className="text-2xl font-semibold mb-2">Event Type Distribution</h2>
                        <Recharts.ResponsiveContainer width="100%" height={300}>
                            <Recharts.BarChart data={typeChartData}>
                                <Recharts.CartesianGrid strokeDasharray="3 3" />
                                <Recharts.XAxis dataKey="type" label={{ value: 'Event Type', position: 'insideBottom', offset: -5 }} />
                                <Recharts.YAxis label={{ value: 'Count', angle: -90, position: 'insideLeft' }} />
                                <Recharts.Tooltip />
                                <Recharts.Legend />
                                <Recharts.Bar dataKey="count" fill="#8884d8" />
                            </Recharts.BarChart>
                        </Recharts.ResponsiveContainer>
                    </div>

                    <div className="chart-container">
                        <h2 className="text-2xl font-semibold mb-2">Top 5 Locations</h2>
                        <Recharts.ResponsiveContainer width="100%" height={300}>
                            <Recharts.BarChart data={locationChartData}>
                                <Recharts.CartesianGrid strokeDasharray="3 3" />
                                <Recharts.XAxis dataKey="location" label={{ value: 'Location', position: 'insideBottom', offset: -5 }} tick={{ fontSize: 12 }} />
                                <Recharts.YAxis label={{ value: 'Count', angle: -90, position: 'insideLeft' }} />
                                <Recharts.Tooltip />
                                <Recharts.Legend />
                                <Recharts.Bar dataKey="count" fill="#82ca9d" />
                            </Recharts.BarChart>
                        </Recharts.ResponsiveContainer>
                    </div>

                    <div className="chart-container">
                        <h2 className="text-2xl font-semibold mb-2">Cluster Distribution</h2>
                        <Recharts.ResponsiveContainer width="100%" height={300}>
                            <Recharts.BarChart data={clusterChartData}>
                                <Recharts.CartesianGrid strokeDasharray="3 3" />
                                <Recharts.XAxis dataKey="cluster" label={{ value: 'Cluster ID', position: 'insideBottom', offset: -5 }} />
                                <Recharts.YAxis label={{ value: 'Count', angle: -90, position: 'insideLeft' }} />
                                <Recharts.Tooltip />
                                <Recharts.Legend />
                                <Recharts.Bar dataKey="count" fill="#ff7300" />
                            </Recharts.BarChart>
                        </Recharts.ResponsiveContainer>
                    </div>

                    <div className="mt-6">
                        <h2 className="text-2xl font-semibold mb-2">Interesting Fact</h2>
                        <p>
                            The location <strong>{{ top_location }}</strong> appears most frequently,
                            with <strong>{{ top_location_count }}</strong> events, indicating a potential parking hotspot.
                        </p>
                    </div>

                    <div className="mt-6">
                        <h2 className="text-2xl font-semibold mb-2">Conclusion</h2>
                        <p>
                            The DBSCAN clustering reveals patterns in vehicle movements, with key locations and event types identified.
                            Outliers (noise points) may indicate anomalies, such as unusually long parking or high-mileage trips.
                            Future enhancements could include spatial coordinates for precise clustering.
                        </p>
                    </div>
                </div>
            );

            const root = ReactDOM.createRoot(document.getElementById('root'));
            root.render(<App />);
        </script>
    </body>
    </html>
    """
    
    # Render the template
    template = Template(html_template)
    html_content = template.render(
        type_chart_data=type_chart_data,
        location_chart_data=location_chart_data,
        cluster_chart_data=cluster_chart_data,
        top_location=list(location_counts.keys())[0],
        top_location_count=list(location_counts.values())[0]
    )
    
    # Save the HTML report
    with open('vehicle_tracking_report.html', 'w') as f:
        f.write(html_content)
    
    return html_content

# Main execution
def main():
    # Fetch data
    df = fetch_data_from_snowflake()
    if df is None:
        print("Failed to fetch data. Exiting.")
        return
    
    # Preprocess data
    df_processed, le_type, le_location = preprocess_data(df)
    
    # Apply DBSCAN
    df_clustered = apply_dbscan(df_processed)
    
    # Generate report
    generate_html_report(df_clustered, le_type, le_location)
    
    print("Report generated: vehicle_tracking_report.html")

if __name__ == "__main__":
    main()