import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
rpm = np.random.uniform(1000, 6000, 100)
fuel_cons = (rpm * 0.005) + np.random.normal(0, 5, 100) + 10
load_status = np.random.choice(["Low Load", "Medium Load", "High Load"], 100)

df = pd.DataFrame({"RPM": rpm, "FuelConsumption": fuel_cons, "EngineLoad": load_status})

df["EngineLoad"] = df["EngineLoad"].astype("category")
print(df["EngineLoad"].value_counts(sort=False, dropna=False))
# Before renaming categories first check order with: df["EngineLoad"].cat.categories
df["EngineLoad"] = df["EngineLoad"].cat.rename_categories(["High", "Low", "Medium"])
print(df.describe(include="all"))
plt.scatter(x="EngineLoad", y="FuelConsumption", data=df, alpha=0.3)
plt.show()
sns.boxplot(x="EngineLoad", y="FuelConsumption", data=df)
plt.show()
sns.countplot(x="EngineLoad", data=df)
plt.show()
# 2. Create an interactive scatter plot
# fig = px.scatter(
#     df,
#     x="RPM",
#     y="FuelConsumption",
#     color="EngineLoad",  # Automatic color grouping
#     trendline="ols",  # Adds an Ordinary Least Squares linear regression line
#     title="Engine Performance: Fuel Consumption vs. RPM",
#     labels={
#         "RPM": "Engine Speed (RPM)",
#         "FuelConsumption": "Fuel Consumption (L/h)",
#     },
#     hover_data=["EngineLoad"],  # Extra data to display on hover
#     template="plotly_dark",  # Clean, dark sleek UI template
# )

# 3. Render the plot in your browser or Jupyter notebook
#fig.show()

# 4. (Optional) Save it as a standalone interactive HTML file
# fig.write_html("engine_report.html")
