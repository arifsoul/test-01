import snowflake.connector

# Membuat koneksi ke Snowflake
conn = snowflake.connector.connect(
    user='TES_USR_LACAK',
    password='StrongPassword123',
    account='hb01677.ap-southeast-3.aws',  # tanpa https:// dan /computing.com/
    # warehouse='YOUR_WAREHOUSE',             # ganti sesuai warehouse-mu
    # database='YOUR_DATABASE',               # ganti sesuai database-mu
    # schema='YOUR_SCHEMA'                    # ganti sesuai schema-mu
)

# Membuat cursor untuk eksekusi query
cursor = conn.cursor()

try:
    # Contoh menjalankan query
    cursor.execute("SELECT CURRENT_VERSION()")
    result = cursor.fetchone()
    print("Snowflake Version:", result[0])

finally:
    # Tutup koneksi
    cursor.close()
    conn.close()
