import pandas as pd
import json

df = pd.read_csv('fintech_mule_dataset.csv')
# Take a representative sample of 250 records for smooth in-browser performance + summary stats
sample_df = df.sample(n=300, random_state=42).copy()
records = sample_df.to_dict(orient='records')

summary_data = {
    "records": records,
    "total_count": len(df),
    "legit_count": int((df['Target'] == 0).sum()),
    "fast_mule_count": int((df['Mule_Type'] == 'Fast Burner').sum()),
    "sleeper_mule_count": int((df['Mule_Type'] == 'Sleeper Mule').sum())
}

with open('dataset_web.json', 'w') as f:
    json.dump(summary_data, f, indent=2)

print("Exported dataset_web.json successfully!")
