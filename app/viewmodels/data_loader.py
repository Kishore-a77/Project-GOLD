from app.db.supabase_client import engine

def get_features_df():
    return pd.read_sql("SELECT * FROM features ORDER BY date", engine)