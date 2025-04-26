import snowflake.connector
import numpy as np
import datetime
import pandas as pd
import re
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from jinja2 import Template
from tqdm import tqdm

# Snowflake connection parameters
conn = snowflake.connector.connect(
    user='TES_USR_LACAK',
    password='StrongPassword123',
    account='hb01677.ap-southeast-3.aws',
    database='TES_DB_RAW',
    schema='RAW_LACAK'
)

# Fetch data from Snowflake
def fetch_data_from_snowflake():
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tr_track")
        results = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        df = pd.DataFrame(results, columns=columns)
        return df
    except Exception as e:
        print(f"Error connecting to Snowflake: {e}")
        return None

# Convert HH:MM:SS to seconds
def duration_to_seconds(duration):
    if not duration:
        return 0
    try:
        h, m, s = map(int, duration.split(':'))
        return h * 3600 + m * 60 + s
    except:
        return 0

# Extract mileage from DESCRIPTION
def extract_mileage(description):
    if not description:
        return 0
    match = re.search(r'Mileage: (\d+\.?\d*)', str(description))
    return float(match.group(1)) if match else 0

# Preprocess and encode
def preprocess_data(data):
    filtered = data[
        data['UNIT'].notnull() &
        data['TYPE'].notnull() &
        data['DATE_TIME_BEGIN'].notnull() &
        data['DATE_TIME_END'].notnull()
    ]
    if filtered.empty:
        print("No valid data after filtering.")
        return None, None, None

    processed = []
    for _, row in tqdm(filtered.iterrows(), total=len(filtered), desc='Preprocessing'):
        try:
            bd = (row['DATE_TIME_BEGIN'].to_pydatetime()
                  if isinstance(row['DATE_TIME_BEGIN'], pd.Timestamp)
                  else datetime.datetime.strptime(row['DATE_TIME_BEGIN'], '%Y-%m-%d %H:%M:%S'))
            ed = (row['DATE_TIME_END'].to_pydatetime()
                  if isinstance(row['DATE_TIME_END'], pd.Timestamp)
                  else datetime.datetime.strptime(row['DATE_TIME_END'], '%Y-%m-%d %H:%M:%S'))
            processed.append({
                'UNIT': row['UNIT'],
                'TYPE': row['TYPE'],
                'INITIAL_LOCATION': row['INITIAL_LOCATION'],
                'FINAL_LOCATION': row['FINAL_LOCATION'],
                'DATE_TIME_BEGIN': bd,
                'DATE_TIME_END': ed,
                'HOUR_OF_DAY': bd.hour,
                'DAY_OF_WEEK': bd.weekday(),
                'DURATION_SECONDS': duration_to_seconds(row['DURATION']),
                'MILEAGE': extract_mileage(row['DESCRIPTION'])
            })
        except Exception as e:
            print(f"Error processing row: {e}")
            continue

    le_t = LabelEncoder()
    le_l = LabelEncoder()
    for row in tqdm(processed, desc='Encoding'):
        row['TYPE_ENCODED'] = le_t.fit_transform([row['TYPE']])[0]
        row['INITIAL_LOCATION_ENCODED'] = le_l.fit_transform([row['INITIAL_LOCATION']])[0]

    return processed, le_t, le_l

# Apply DBSCAN clustering
def apply_dbscan(data):
    if not data:
        return None
    features = [
        [r['DURATION_SECONDS'], r['MILEAGE'], r['HOUR_OF_DAY'], r['DAY_OF_WEEK'],
         r['TYPE_ENCODED'], r['INITIAL_LOCATION_ENCODED']]
        for r in data
    ]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    db = DBSCAN(eps=0.5, min_samples=5)
    clusters = db.fit_predict(scaled)
    for i, row in enumerate(data):
        row['CLUSTER'] = clusters[i]
    return data

# Generate HTML report with progress bars and correct Jinja raw blocks
def generate_html_report(data, le_type, le_location):
    if not data:
        print("No data to generate report.")
        return None

    # Aggregate with tqdm
    type_counts, loc_counts, clust_counts = {}, {}, {}
    for row in tqdm(data, desc='Aggregating'):
        type_counts[row['TYPE']] = type_counts.get(row['TYPE'], 0) + 1
        loc_counts[row['INITIAL_LOCATION']] = loc_counts.get(row['INITIAL_LOCATION'], 0) + 1
        clust_counts[row['CLUSTER']] = clust_counts.get(row['CLUSTER'], 0) + 1

    type_chart = [{'type': k, 'count': v} for k, v in type_counts.items()]
    loc_chart = [{'location': k, 'count': v} for k, v in sorted(loc_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
    clust_chart = [{'cluster': str(k), 'count': v} for k, v in clust_counts.items() if k != -1]

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
        <style> body { font-family: Arial; } .chart-container { margin:20px 0; } </style>
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
                <div className="chart-container">
                  <h2 className="text-2xl font-semibold mb-2">Event Type Distribution</h2>
                  <Recharts.ResponsiveContainer width="100%" height={300}>
                    <Recharts.BarChart data={typeChartData}>
                      <Recharts.CartesianGrid strokeDasharray="3 3" />
                      <Recharts.XAxis dataKey="type" label={% raw %}{ value: 'Event Type', position: 'insideBottom', offset: -5 }{% endraw %} />
                      <Recharts.YAxis label={% raw %}{ value: 'Count', angle: -90, position: 'insideLeft' }{% endraw %} />
                      <Recharts.Tooltip />
                      <Recharts.Legend />
                      <Recharts.Bar dataKey="count" fill="#8884d8" />
                    </Recharts.BarChart>
                  </Recharts.ResponsiveContainer>
                </div>
                <!-- similar blocks for location & cluster with raw -->
              </div>
            );
            const root = ReactDOM.createRoot(document.getElementById('root'));
            root.render(<App />);
        </script>
    </body>
    </html>
    """
    
    template = Template(html_template)
    html = template.render(
        type_chart_data=type_chart,
        location_chart_data=loc_chart,
        cluster_chart_data=clust_chart
    )
    with open('vehicle_tracking_report.html', 'w') as f:
        f.write(html)
    return html

# Main
if __name__ == '__main__':
    df = fetch_data_from_snowflake()
    if df is None: exit()
    proc, lt, ll = preprocess_data(df)
    if proc is None: exit()
    clustered = apply_dbscan(proc)
    if clustered is None: exit()
    print("Generating HTML report...")
    generate_html_report(clustered, lt, ll)
    print("Done: vehicle_tracking_report.html")
