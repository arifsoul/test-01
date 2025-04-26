import snowflake.connector
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from jinja2 import Template
import datetime
import pandas as pd
import re

# Snowflake connection parameters
conn = snowflake.connector.connect(
    user='TES_USR_LACAK',
    password='StrongPassword123',
    account='hb01677.ap-southeast-3.aws',
    database='TES_DB_RAW',          # database dari query kamu
    schema='RAW_LACAK'              # schema dari query kamu
)

# Function to fetch data from Snowflake without pandas
def fetch_data_from_snowflake():
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM tr_track"
        cursor.execute(query)
        # Fetch column names
        results = cursor.fetchall()
        # Ambil nama kolom
        columns = [col[0] for col in cursor.description]

        # Buat dataframe biar rapi tampilannya
        df = pd.DataFrame(results, columns=columns)
        
        return df
    except Exception as e:
        print(f"Error connecting to Snowflake: {e}")
        return None

# Function to convert duration (HH:MM:SS) to seconds
def duration_to_seconds(duration):
    if not duration:
        return 0
    try:
        h, m, s = map(int, duration.split(':'))
        return h * 3600 + m * 60 + s
    except:
        return 0

# Function to extract mileage from DESCRIPTION
def extract_mileage(description):
    if not description:
        return 0
    match = re.search(r'Mileage: (\d+\.?\d*)', str(description))
    return float(match.group(1)) if match else 0

# Preprocessing and feature engineering
def preprocess_data(data):
    # Filter out invalid rows
    filtered_data = data[
        data['UNIT'].notnull() & 
        data['TYPE'].notnull() & 
        data['DATE_TIME_BEGIN'].notnull() & 
        data['DATE_TIME_END'].notnull()
    ]
    
    if filtered_data.empty:
        print("No valid data after filtering.")
        return None, None, None
    
    # Parse dates and extract features
    processed_data = []
    for _, row in filtered_data.iterrows():
        try:
            # Convert Timestamp to datetime or use directly
            begin_date = row['DATE_TIME_BEGIN'].to_pydatetime() if isinstance(row['DATE_TIME_BEGIN'], pd.Timestamp) else datetime.datetime.strptime(row['DATE_TIME_BEGIN'], '%Y-%m-%d %H:%M:%S')
            end_date = row['DATE_TIME_END'].to_pydatetime() if isinstance(row['DATE_TIME_END'], pd.Timestamp) else datetime.datetime.strptime(row['DATE_TIME_END'], '%Y-%m-%d %H:%M:%S')
            
            processed_row = {
                'UNIT': row['UNIT'],
                'TYPE': row['TYPE'],
                'INITIAL_LOCATION': row['INITIAL_LOCATION'],
                'FINAL_LOCATION': row['FINAL_LOCATION'],
                'DATE_TIME_BEGIN': begin_date,
                'DATE_TIME_END': end_date,
                'HOUR_OF_DAY': begin_date.hour,
                'DAY_OF_WEEK': begin_date.weekday(),
                'DURATION_SECONDS': duration_to_seconds(row['DURATION']),
                'MILEAGE': extract_mileage(row['DESCRIPTION'])
            }
            processed_data.append(processed_row)
        except Exception as e:
            print(f"Error processing row {row}: {e}")
            continue
    
    # Encode categorical variables
    types = [row['TYPE'] for row in processed_data]
    locations = [row['INITIAL_LOCATION'] for row in processed_data]
    le_type = LabelEncoder()
    le_location = LabelEncoder()
    for row in processed_data:
        row['TYPE_ENCODED'] = le_type.fit_transform([row['TYPE']])[0]
        row['INITIAL_LOCATION_ENCODED'] = le_location.fit_transform([row['INITIAL_LOCATION']])[0]
    
    return processed_data, le_type, le_location

# Apply DBSCAN clustering
def apply_dbscan(data):
    if not data:
        return None
    features = [
        [row['DURATION_SECONDS'], row['MILEAGE'], row['HOUR_OF_DAY'], row['DAY_OF_WEEK'],
         row['TYPE_ENCODED'], row['INITIAL_LOCATION_ENCODED']]
        for row in data
    ]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(scaled_features)
    for i, row in enumerate(data):
        row['CLUSTER'] = clusters[i]
    return data

# Generate HTML report with Recharts
def generate_html_report(data, le_type, le_location):
    if not data:
        print("No data to generate report.")
        return None
    
    # Aggregate data for visualizations
    type_counts = {}
    location_counts = {}
    cluster_counts = {}
    for row in data:
        type_counts[row['TYPE']] = type_counts.get(row['TYPE'], 0) + 1
        location_counts[row['INITIAL_LOCATION']] = location_counts.get(row['INITIAL_LOCATION'], 0) + 1
        cluster_counts[row['CLUSTER']] = cluster_counts.get(row['CLUSTER'], 0) + 1
    
    type_chart_data = [{'type': k, 'count': v} for k, v in type_counts.items()]
    location_chart_data = [
        {'location': k, 'count': v} for k, v in
        sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    ]
    cluster_chart_data = [{'cluster': str(k), 'count': v} for k, v in cluster_counts.items() if k != -1]
    
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
            const typeChartData = {{ type_chart_data | tojson | safe }};
            const locationChartData = {{ location_chart_data | tojson | safe }};
            const clusterChartData = {{ cluster_chart_data | tojson | safe }};
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
    template = Template(html_template)
    html_content = template.render(
        type_chart_data=type_chart_data,
        location_chart_data=location_chart_data,
        cluster_chart_data=cluster_chart_data,
        top_location=location_chart_data[0]['location'] if location_chart_data else 'Unknown',
        top_location_count=location_chart_data[0]['count'] if location_chart_data else 0
    )
    
    with open('vehicle_tracking_report.html', 'w') as f:
        f.write(html_content)
    return html_content

# Main execution
def main():
    data = fetch_data_from_snowflake()
    if data is None:
        print("Failed to fetch data. Exiting.")
        return
    processed_data, le_type, le_location = preprocess_data(data)
    if processed_data is None:
        print("Failed to preprocess data. Exiting.")
        return
    clustered_data = apply_dbscan(processed_data)
    if clustered_data is None:
        print("Failed to apply clustering. Exiting.")
        return
    generate_html_report(clustered_data, le_type, le_location)
    print("Report generated: vehicle_tracking_report.html")

if __name__ == "__main__":
    main()