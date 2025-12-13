from app.viewmodels.snowflake_data_loader import load_processed_data

df = load_processed_data()

print("Columns:", df.columns.tolist())
print(df.head())
