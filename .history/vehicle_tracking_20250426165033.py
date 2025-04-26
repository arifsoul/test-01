import snowflake.connector
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
import datetime
import pandas as pd
import re
import json
from tqdm.auto import tqdm

# Snowflake connection parameters
conn = snowflake.connector.connect(
    user='TES_USR_LACAK',
    password='StrongPassword123',
    account='hb01677.ap-southeast-3.aws',
    database='TES_DB_RAW',          # database from your query
    schema='RAW_LACAK'              # schema from your query
)

# Function to fetch data from Snowflake with progress feedback
def fetch_data_from_snowflake():
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM tr_track"
        cursor.execute(query)
        # Fetch column names
        columns = [col[0] for col in cursor.description]
        results = cursor.fetchall()

        print("Building DataFrame from Snowflake results...")
        df = pd.DataFrame(results, columns=columns)
        return df
    except Exception as e:
        print(f"Error connecting to Snowflake: {e}")
        return None

# Convert duration (HH:MM:SS) to seconds
def duration_to_seconds(duration):
    if not duration:
        return 0
    try:
        h, m, s = map(int, duration.split(':'))
        return h * 3600 + m * 60 + s
    except:
        return 0

# Extract mileage from DESCRIPTION text
def extract_mileage(description):
    if not description:
        return 0
    match = re.search(r'Mileage: (\d+\.?\d*)', str(description))
    return float(match.group(1)) if match else 0

# Preprocessing and feature engineering with tqdm progress bar
def preprocess_data(df):
    # Filter out invalid rows
    filtered = df[
        df['UNIT'].notnull() &
        df['TYPE'].notnull() &
        df['DATE_TIME_BEGIN'].notnull() &
        df['DATE_TIME_END'].notnull()
    ]
    if filtered.empty:
        print("No valid data after filtering.")
        return None, None, None

    processed = []
    print("Preprocessing rows...")
    for _, row in tqdm(filtered.iterrows(), total=len(filtered), desc="Preprocessing"):
        try:
            begin = row['DATE_TIME_BEGIN']
            end = row['DATE_TIME_END']
            # Ensure datetime
            if isinstance(begin, pd.Timestamp):
                begin = begin.to_pydatetime()
            else:
                begin = datetime.datetime.strptime(str(begin), '%Y-%m-%d %H:%M:%S')
            if isinstance(end, pd.Timestamp):
                end = end.to_pydatetime()
            else:
                end = datetime.datetime.strptime(str(end), '%Y-%m-%d %H:%M:%S')

            processed.append({
                'UNIT': row['UNIT'],
                'TYPE': row['TYPE'],
                'INITIAL_LOCATION': row['INITIAL_LOCATION'],
                'FINAL_LOCATION': row['FINAL_LOCATION'],
                'DATE_TIME_BEGIN': begin,
                'DATE_TIME_END': end,
                'HOUR_OF_DAY': begin.hour,
                'DAY_OF_WEEK': begin.weekday(),
                'DURATION_SECONDS': duration_to_seconds(row['DURATION']),
                'MILEAGE': extract_mileage(row['DESCRIPTION'])
            })
        except Exception as e:
            print(f"Error processing row: {e}")
            continue

    # Encode categorical variables
    types = [r['TYPE'] for r in processed]
    locs = [r['INITIAL_LOCATION'] for r in processed]
    le_type = LabelEncoder().fit(types)
    le_loc = LabelEncoder().fit(locs)
    for r in processed:
        r['TYPE_ENCODED'] = int(le_type.transform([r['TYPE']])[0])
        r['INITIAL_LOCATION_ENCODED'] = int(le_loc.transform([r['INITIAL_LOCATION']])[0])

    return processed, le_type, le_loc

# Apply DBSCAN clustering with tqdm progress bar
def apply_dbscan(data):
    if not data:
        return None

    features = [[
        r['DURATION_SECONDS'], r['MILEAGE'], r['HOUR_OF_DAY'], r['DAY_OF_WEEK'],
        r['TYPE_ENCODED'], r['INITIAL_LOCATION_ENCODED']
    ] for r in data]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    print("Running DBSCAN clustering...")
    db = DBSCAN(eps=0.5, min_samples=5)
    clusters = db.fit_predict(scaled)

    print("Assigning cluster labels...")
    for r, c in tqdm(zip(data, clusters), total=len(data), desc="Clustering"):
        r['CLUSTER'] = int(c)
    return data

# Generate static HTML report file with embedded JSON data for Recharts
def generate_html_report(data, output_file='vehicle_tracking_report.html'):
    if not data:
        print("No data to generate report.")
        return

    # Aggregate counts
    type_counts = {}
    loc_counts = {}
    cluster_counts = {}
    for r in data:
        type_counts[r['TYPE']] = type_counts.get(r['TYPE'], 0) + 1
        loc_counts[r['INITIAL_LOCATION']] = loc_counts.get(r['INITIAL_LOCATION'], 0) + 1
        c = r['CLUSTER']
        if c != -1:
            cluster_counts[c] = cluster_counts.get(c, 0) + 1

    type_chart = [{'type': k, 'count': v} for k, v in type_counts.items()]
    loc_chart = sorted([
        {'location': k, 'count': v} for k, v in loc_counts.items()
    ], key=lambda x: x['count'], reverse=True)[:5]
    cluster_chart = [{'cluster': str(k), 'count': v} for k, v in cluster_counts.items()]
    top_loc, top_count = (loc_chart[0]['location'], loc_chart[0]['count']) if loc_chart else ('Unknown', 0)

    # JSON strings for embedding
    tcd = json.dumps(type_chart)
    lcd = json.dumps(loc_chart)
    ccd = json.dumps(cluster_chart)

    # Full HTML content
    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Vehicle Tracking Analysis Report</title>
    <script src=\"https://cdnjs.cloudflare.com/ajax/libs/prop-types/15.8.1/prop-types.min.js\"></script>
    <script src=\"https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js\"></script>
    <script src=\"https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js\"></script>
    <script src=\"https://cdnjs.cloudflare.com/ajax/libs/recharts/2.15.0/Recharts.min.js\"></script>
    <script src=\"https://cdn.tailwindcss.com\"></script>
    <style>body {{ font-family: Arial, sans-serif; }} .chart-container {{ margin: 20px 0; }}</style>
</head>
<body class=\"bg-gray-100 p-6\">
    <div id=\"root\"></div>
    <script>
        const typeChartData = {tcd};
        const locationChartData = {lcd};
        const clusterChartData = {ccd};
        const App = () => (
            <div className=\"max-w-4xl mx-auto bg-white p-6 rounded-lg shadow-md\">
                <h1 className=\"text-3xl font-bold mb-4\">Vehicle Tracking Analysis Report</h1>
                <p className=\"mb-4\">
                    This report analyzes vehicle tracking data using DBSCAN clustering to identify patterns
                    in trips and parking events. Key insights include frequent locations, event types,
                    and cluster distributions.
                </p>
                <div className=\"chart-container\">
                    <h2 className=\"text-2xl font-semibold mb-2\">Event Type Distribution</h2>
                    <Recharts.ResponsiveContainer width=\"100%\" height={300}>
                        <Recharts.BarChart data={typeChartData}>
                            <Recharts.CartesianGrid strokeDasharray=\"3 3\" />
                            <Recharts.XAxis dataKey=\"type\" label={{ value: 'Event Type', position: 'insideBottom', offset: -5 }} />
                            <Recharts.YAxis label={{ value: 'Count', angle: -90, position: 'insideLeft' }} />
                            <Recharts.Tooltip />
                            <Recharts.Legend />
                            <Recharts.Bar dataKey=\"count\" fill=\"#8884d8\" />
                        </Recharts.BarChart>
                    </Recharts.ResponsiveContainer>
                </div>
                <div className=\"chart-container\">
                    <h2 className=\"text-2xl font-semibold mb-2\">Top 5 Locations</h2>
                    <Recharts.ResponsiveContainer width=\"100%\" height={300}>
                        <Recharts.BarChart data={locationChartData}>
                            <Recharts.CartesianGrid strokeDasharray=\"3 3\" />
                            <Recharts.XAxis dataKey=\"location\" tick={{ fontSize: 12 }} label={{ value: 'Location', position: 'insideBottom', offset: -5 }} />
                            <Recharts.YAxis label={{ value: 'Count', angle: -90, position: 'insideLeft' }} />
                            <Recharts.Tooltip />
                            <Recharts.Legend />
                            <Recharts.Bar dataKey=\"count\" fill=\"#82ca9d\" />
                        </Recharts.BarChart>
                    </Recharts.ResponsiveContainer>
                </div>
                <div className=\"chart-container\">
                    <h2 className=\"text-2xl font-semibold mb-2\">Cluster Distribution</h2>
                    <Recharts.ResponsiveContainer width=\"100%\" height={300}>
                        <Recharts.BarChart data={clusterChartData}>
                            <Recharts.CartesianGrid strokeDasharray=\"3 3\" />
                            <Recharts.XAxis dataKey=\"cluster\" label={{ value: 'Cluster ID', position: 'insideBottom', offset: -5 }} />
                            <Recharts.YAxis label={{ value: 'Count', angle: -90, position: 'insideLeft' }} />
                            <Recharts.Tooltip />
                            <Recharts.Legend />
                            <Recharts.Bar dataKey=\"count\" fill=\"#ff7300\" />
                        </Recharts.BarChart>
                    </Recharts.ResponsiveContainer>
                </div>
                <div className=\"mt-6\">
                    <h2 className=\"text-2xl font-semibold mb-2\">Interesting Fact</h2>
                    <p>
                        The location <strong>{top_loc}</strong> appears most frequently,
                        with <strong>{top_count}</strong> events, indicating a potential parking hotspot.
                    </p>
                </div>
                <div className=\"mt-6\">
                    <h2 className=\"text-2xl font-semibold mb-2\">Conclusion</h2>
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
    with open(output_file, 'w') as f:
        f.write(html)
    print(f"Report generated: {output_file}")

# Main execution
if __name__ == '__main__':
    df = fetch_data_from_snowflake()
    if df is None:
        exit(1)
    processed, le_type, le_loc = preprocess_data(df)
    if processed is None:
        exit(1)
    clustered = apply_dbscan(processed)
    generate_html_report(clustered)
