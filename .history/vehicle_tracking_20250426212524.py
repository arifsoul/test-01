import snowflake.connector
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
import datetime
import pandas as pd
import re
import html
import json

# Snowflake connection parameters
conn = snowflake.connector.connect(
    user="TES_USR_LACAK",
    password="StrongPassword123",
    account="hb01677.ap-southeast-3.aws",
    database="TES_DB_RAW",
    schema="RAW_LACAK"
)

# Function to fetch data from Snowflake
def fetch_data_from_snowflake():
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM tr_track"
        cursor.execute(query)
        columns = [col[0] for col in cursor.description]
        results = cursor.fetchall()
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
        h, m, s = map(int, duration.split(":"))
        return h * 3600 + m * 60 + s
    except:
        return 0

# Extract mileage from DESCRIPTION
def extract_mileage(description):
    if not description:
        return 0
    match = re.search(r"Mileage: (\d+\.?\d*)", str(description))
    return float(match.group(1)) if match else 0

# Preprocessing and feature engineering
def preprocess_data(data):
    filtered_data = data[
        data["UNIT"].notnull() & 
        data["TYPE"].notnull() & 
        data["DATE_TIME_BEGIN"].notnull() & 
        data["DATE_TIME_END"].notnull()
    ]
    
    if filtered_data.empty:
        print("No valid data after filtering.")
        return None, None, None
    
    processed_data = []
    for _, row in filtered_data.iterrows():
        try:
            begin_date = row["DATE_TIME_BEGIN"].to_pydatetime() if isinstance(row["DATE_TIME_BEGIN"], pd.Timestamp) else datetime.datetime.strptime(row["DATE_TIME_BEGIN"], "%Y-%m-%d %H:%M:%S")
            end_date = row["DATE_TIME_END"].to_pydatetime() if isinstance(row["DATE_TIME_END"], pd.Timestamp) else datetime.datetime.strptime(row["DATE_TIME_END"], "%Y-%m-%d %H:%M:%S")
            
            processed_row = {
                "UNIT": row["UNIT"],
                "TYPE": row["TYPE"],
                "INITIAL_LOCATION": row["INITIAL_LOCATION"],
                "FINAL_LOCATION": row["FINAL_LOCATION"],
                "DATE_TIME_BEGIN": begin_date,
                "DATE_TIME_END": end_date,
                "HOUR_OF_DAY": begin_date.hour,
                "DAY_OF_WEEK": begin_date.weekday(),
                "DURATION_SECONDS": duration_to_seconds(row["DURATION"]),
                "MILEAGE": extract_mileage(row["DESCRIPTION"])
            }
            processed_data.append(processed_row)
        except Exception as e:
            print(f"Error processing row {row}: {e}")
            continue
    
    types = [row["TYPE"] for row in processed_data]
    locations = [row["INITIAL_LOCATION"] for row in processed_data]
    le_type = LabelEncoder()
    le_location = LabelEncoder()
    for row in processed_data:
        row["TYPE_ENCODED"] = le_type.fit_transform([row["TYPE"]])[0]
        row["INITIAL_LOCATION_ENCODED"] = le_location.fit_transform([row["INITIAL_LOCATION"]])[0]
    
    return processed_data, le_type, le_location

# Apply DBSCAN clustering
def apply_dbscan(data):
    if not data:
        return None
    features = [
        [row["DURATION_SECONDS"], row["MILEAGE"], row["HOUR_OF_DAY"], row["DAY_OF_WEEK"],
         row["TYPE_ENCODED"], row["INITIAL_LOCATION_ENCODED"]]
        for row in data
    ]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(scaled_features)
    for i, row in enumerate(data):
        row["CLUSTER"] = clusters[i]
    return data

# Generate HTML report with plain JavaScript
def generate_html_report(data, le_type, le_location):
    if not data:
        print("No data to generate report.")
        return None
    
    # Clean strings to prevent HTML/JavaScript issues
    def clean_string(s):
        if s is None:
            return ""
        if not isinstance(s, str):
            s = str(s)
        # Remove control characters, preserve quotes for JSON
        s = "".join(c for c in s if c.isprintable())
        return html.escape(s, quote=True)
    
    # Aggregate data for visualizations
    type_counts = {}
    location_counts = {}
    cluster_counts = {}
    for row in data:
        type_counts[row["TYPE"]] = type_counts.get(row["TYPE"], 0) + 1
        location_counts[row["INITIAL_LOCATION"]] = location_counts.get(row["INITIAL_LOCATION"], 0) + 1
        cluster_counts[row["CLUSTER"]] = cluster_counts.get(row["CLUSTER"], 0) + 1
    
    type_chart_data = [{"type": clean_string(k), "count": v} for k, v in type_counts.items()]
    location_chart_data = [
        {"location": clean_string(k), "count": v} for k, v in
        sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    ]
    cluster_chart_data = [{"cluster": clean_string(str(k)), "count": v} for k, v in cluster_counts.items() if k != -1]
    
    # Debug: Print data to inspect for issues
    print("type_chart_data:", type_chart_data)
    print("location_chart_data:", location_chart_data)
    print("cluster_chart_data:", cluster_chart_data)
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vehicle Tracking Analysis Report</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/recharts/2.12.7/Recharts.min.js"></script>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .chart-container {{ margin: 20px 0; }}
            .error {{ color: red; font-weight: bold; }}
            .chart {{ width: 100%; height: 300px; }}
        </style>
    </head>
    <body class="bg-gray-100 p-6">
        <div id="root" class="max-w-4xl mx-auto bg-white p-6 rounded-lg shadow-md"></div>
        <div id="error" class="error" style="display: none;"></div>
        <script>
            // Data for charts
            const typeChartData = {type_chart_data};
            const locationChartData = {location_chart_data};
            const clusterChartData = {cluster_chart_data};

            // Error handling function
            function showError(message) {{
                const errorDiv = document.getElementById("error");
                errorDiv.style.display = "block";
                errorDiv.textContent = message;
            }}

            // Function to create a bar chart
            function createBarChart(containerId, data, xKey, fillColor) {{
                try {{
                    const {{ BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer }} = window.Recharts;
                    const chartContainer = document.getElementById(containerId);
                    const chart = new ResponsiveContainer({{ width: "100%", height: 300 }});
                    const barChart = new BarChart({{
                        data: data,
                        children: [
                            new CartesianGrid({{ strokeDasharray: "3 3" }}),
                            new XAxis({{ dataKey: xKey, tick: {{ fontSize: 12 }} }}),
                            new YAxis(),
                            new Tooltip(),
                            new Legend(),
                            new Bar({{ dataKey: "count", fill: fillColor }})
                        ]
                    }});
                    chart.setContainer(chartContainer);
                    chart.add(barChart);
                    chart.render();
                }} catch (e) {{
                    showError(`Error rendering chart: ${{e.toString()}}`);
                }}
            }}

            // Main function to build the report
            function renderReport() {{
                try {{
                    const root = document.getElementById("root");
                    
                    // Main container
                    root.innerHTML = `
                        <h1 class="text-3xl font-bold mb-4">Vehicle Tracking Analysis Report</h1>
                        <p class="mb-4">
                            This report analyzes vehicle tracking data using DBSCAN clustering to identify patterns
                            in trips and parking events. Key insights include frequent locations, event types,
                            and cluster distributions.
                        </p>
                        <div id="type-chart" class="chart-container">
                            <h2 class="text-2xl font-semibold mb-2">Event Type Distribution</h2>
                            <div id="type-chart-canvas" class="chart"></div>
                        </div>
                        <div id="location-chart" class="chart-container">
                            <h2 class="text-2xl font-semibold mb-2">Top 5 Locations</h2>
                            <div id="location-chart-canvas" class="chart"></div>
                        </div>
                        <div id="cluster-chart" class="chart-container">
                            <h2 class="text-2xl font-semibold mb-2">Cluster Distribution</h2>
                            <div id="cluster-chart-canvas" class="chart"></div>
                        </div>
                        <div class="mt-6">
                            <h2 class="text-2xl font-semibold mb-2">Interesting Fact</h2>
                            <p>
                                The location <strong>${{locationChartData[0].location}}</strong> appears most frequently,
                                with <strong>${{locationChartData[0].count}}</strong> events, indicating a potential parking hotspot.
                            </p>
                        </div>
                        <div class="mt-6">
                            <h2 class="text-2xl font-semibold mb-2">Conclusion</h2>
                            <p>
                                The DBSCAN clustering reveals patterns in vehicle movements, with key locations and event types identified.
                                Outliers (noise points) may indicate anomalies, such as unusually long parking or high-mileage trips.
                                Future enhancements could include spatial coordinates for precise clustering.
                            </p>
                        </div>
                    `;

                    // Render charts
                    createBarChart("type-chart-canvas", typeChartData, "type", "#8884d8");
                    createBarChart("location-chart-canvas", locationChartData, "location", "#82ca9d");
                    createBarChart("cluster-chart-canvas", clusterChartData, "cluster", "#ff7300");
                }} catch (e) {{
                    showError(`Error rendering application: ${{e.toString()}}`);
                }}
            }}

            // Execute rendering
            document.addEventListener("DOMContentLoaded", renderReport);
        </script>
    </body>
    </html>
    """
    
    with open("vehicle_tracking_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("Report generated: vehicle_tracking_report.html")
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

if __name__ == "__main__":
    main()