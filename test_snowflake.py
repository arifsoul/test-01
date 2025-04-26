import snowflake.connector
import pandas as pd

# Membuat koneksi ke Snowflake
conn = snowflake.connector.connect(
    user='TES_USR_LACAK',
    password='StrongPassword123',
    account='hb01677.ap-southeast-3.aws',
    database='TES_DB_RAW',          # database dari query kamu
    schema='RAW_LACAK'              # schema dari query kamu
)

# Membuat cursor untuk eksekusi query
cursor = conn.cursor()

try:
    # Eksekusi query
    query = "SELECT * FROM tr_track"
    cursor.execute(query)
    
    # Ambil semua data
    results = cursor.fetchall()

    # Ambil nama kolom
    columns = [col[0] for col in cursor.description]

    # Buat dataframe biar rapi tampilannya
    df = pd.DataFrame(results, columns=columns)

    print(df)

finally:
    cursor.close()
    conn.close()
