import pandas as pd
import numpy as np

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# LOAD & CLEAN Dataset
df = pd.read_csv("fastfood.csv")
df.columns = df.columns.str.strip()
df["restaurant"] = df["restaurant"].str.strip()
df["item"] = df["item"].str.strip()


df["Food Type"] = np.where(df["item"].str.contains("salad", case=False), "Salad", "Other")

# Exact duplicate rows are removed
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Rows: {before} -> {len(df)} after removing exact duplicates | "
      f"Restaurants: {df['restaurant'].nunique()}")
print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0].to_string())

# min-max scaler (0-1)
mm = lambda s: (s - s.min()) / (s.max() - s.min())

def title(t):
    print("\n" + "=" * len(t) + f"\n{t}\n" + "=" * len(t))

# Q1: Which restaurant has the highest number of food items in the dataset?

title("Q1. Restaurant with the most food items")
q1 = df["restaurant"].value_counts()
print(q1)
print(f"\nANSWER: {q1.index[0]} with {q1.iloc[0]} items")

# Q2: Which food items have the highest number of calories? Find the top 10 items.
title("Q2. Top 10 items by calories")
print(df.nlargest(10, "calories")[["restaurant", "item", "calories"]].to_string(index=False))

# Q3: What is the average calorie count of food items for each restaurant?
title("Q3. Average calories per restaurant")
q3 = df.groupby("restaurant")["calories"].mean().round(2).sort_values(ascending=False)
print(q3)

# Q4: Which restaurant offers food items with the highest average protein content?
title("Q4. Restaurant with the highest average protein")
q4 = df.groupby("restaurant")["protein"].mean().round(2).sort_values(ascending=False)
print(q4)
print(f"\nANSWER: {q4.index[0]} ({q4.iloc[0]} g)")

# Q5: Which restaurant has the highest average amount of total fat, and which restaurant has the lowest?
title("Q5. Highest and lowest average total fat")
q5 = df.groupby("restaurant")["total_fat"].mean().round(2).sort_values(ascending=False)
print(q5)
print(f"\nHIGHEST: {q5.index[0]} ({q5.iloc[0]} g) | LOWEST: {q5.index[-1]} ({q5.iloc[-1]} g)")

# Q6: Which food items contain the highest amount of sugar, and how many calories do those items contain?
title("Q6. Items with the most sugar and their calories")
print(df.nlargest(10, "sugar")[["restaurant", "item", "sugar", "calories"]].to_string(index=False))

# Q7: How does the type of food item (salad or other) affect its average calories, fat, carbohydrates, and protein?
title("Q7. Salad vs Other: average calories, fat, carbs, protein")
q7 = (df.groupby("Food Type")
        .agg(Count=("item", "size"),
             Avg_Calories=("calories", "mean"),
             Avg_Total_Fat=("total_fat", "mean"),
             Avg_Carbs=("total_carb", "mean"),
             Avg_Protein=("protein", "mean")).round(2))
print(q7)

# Q8: Which restaurant offers the best combination of low calories and high protein? Use the available nutritional information to support your answer.
title("Q8. Best low-calorie + high-protein restaurant")
q8 = df.groupby("restaurant").agg(Avg_Calories=("calories", "mean"),
                                  Avg_Protein=("protein", "mean")).round(2)
q8["Protein_per_100_Cal"] = (df.groupby("restaurant")["protein"].sum()
                             / df.groupby("restaurant")["calories"].sum() * 100).round(2)

q8["Score"] = (0.5 * mm(q8["Avg_Protein"]) + 0.5 * (1 - mm(q8["Avg_Calories"]))).round(3)
q8 = q8.sort_values("Score", ascending=False)
print(q8)
print(f"\nANSWER: {q8.index[0]}  (best protein-per-calorie: "
      f"{q8['Protein_per_100_Cal'].idxmax()})")

# Q9: Can you identify the 10 least healthy food items by creating a Health Risk Score using calories, total fat, saturated fat, trans fat, cholesterol, and sodium?
title("Q9. 10 least healthy items (Health Risk Score)")
risk_cols = ["calories", "total_fat", "sat_fat", "trans_fat", "cholesterol", "sodium"]

rw = {"calories": 0.20, "total_fat": 0.15, "sat_fat": 0.20,
      "trans_fat": 0.15, "cholesterol": 0.15, "sodium": 0.15}
df["Health_Risk_Score"] = sum(mm(df[c]) * w for c, w in rw.items()) * 100
print("Score (0-100) = 100 * (0.20*calories + 0.15*total_fat + 0.20*sat_fat + "
      "0.15*trans_fat + 0.15*cholesterol + 0.15*sodium), each min-max scaled")
print(df.nlargest(10, "Health_Risk_Score")
        [["restaurant", "item"] + risk_cols + ["Health_Risk_Score"]]
        .round(1).to_string(index=False))

# Q10: Can you compare the nutritional profiles of all restaurants and determine which restaurant provides the healthiest overall menu? Create your own scoring system using calories, fat, sugar, sodium, fiber, protein, and vitamins, and justify your ranking?
title("Q10. Healthiest overall restaurant menu")
prof = df.groupby("restaurant")[["calories", "total_fat", "sugar", "sodium",
                                 "fiber", "protein", "vit_a", "vit_c", "calcium"]].mean()

prof["vitamins"] = prof[["vit_a", "vit_c", "calcium"]].mean(axis=1)
print("Average nutrition per restaurant:")
print(prof.round(2))

bad  = ["calories", "total_fat", "sugar", "sodium"]      # lower is better
good = ["fiber", "protein", "vitamins"]                  # higher is better
w10 = {"calories": 0.20, "total_fat": 0.15, "sugar": 0.10, "sodium": 0.20,
       "fiber": 0.10, "protein": 0.15, "vitamins": 0.10}
score = pd.DataFrame(index=prof.index)
for c in bad:
    score[c] = 1 - mm(prof[c])
for c in good:
    score[c] = mm(prof[c])
wts = pd.Series(w10)
avail = score.notna()
prof["Healthiness_Score"] = ((score.fillna(0) * wts).sum(axis=1)
                             / (avail * wts).sum(axis=1) * 100).round(1)
print("\nScore (0-100): lower-is-better nutrients are inverted, all min-max scaled across restaurants.")
print("Missing vitamin data (Burger King) -> weights re-scaled over the available nutrients.")
print("Weights:", w10)
rank = prof.sort_values("Healthiness_Score", ascending=False)
print("\nRanking:")
print(rank[["Healthiness_Score"]].assign(Rank=range(1, len(rank) + 1)).to_string())
print("\nScore contribution by nutrient (weighted, out of 100):")
print((score * wts * 100).loc[rank.index].round(1))
print(f"\nANSWER: {rank.index[0]} has the healthiest overall menu "
      f"({rank['Healthiness_Score'].iloc[0]}/100); least healthy: "
      f"{rank.index[-1]} ({rank['Healthiness_Score'].iloc[-1]}/100)")