import pandas as pd

df = pd.read_csv("C:/Users/yuzhi/Desktop/month 3/DETECCION DE FRACTURAS.v4i.multiclass/train/_classes.csv")  # your CSV
num_pos = df["Fractura"].sum()
num_neg = len(df) - num_pos

pos_weight = num_neg / num_pos
print("Positive weight:", pos_weight)
