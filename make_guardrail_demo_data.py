import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "expenses.csv"
OUTPUT_PATH = BASE_DIR / "data" / "expenses_guardrail_demo.csv"

df = pd.read_csv(DATA_PATH).drop_duplicates()

np.random.seed(42)

race_categories = ["Black", "Latino", "White", "Asian", "Other"]
race_probs = [0.22, 0.18, 0.45, 0.10, 0.05]

df["race"] = np.random.choice(
    race_categories,
    size=len(df),
    p=race_probs
)

zip_by_region = {
    "northeast": ["10001", "02108", "19103", "11201"],
    "northwest": ["98101", "97201", "83702", "99201"],
    "southeast": ["30303", "33101", "37201", "70112"],
    "southwest": ["85001", "75201", "78701", "89101"]
}

def assign_zip(region):
    region = str(region).lower()
    return np.random.choice(zip_by_region.get(region, ["60601"]))

df["zipcode"] = df["region"].apply(assign_zip)

df.to_csv(OUTPUT_PATH, index=False)

print(f"Created {OUTPUT_PATH}")
print(df.head())