import numpy as np
import pandas as pd
import plotly.express as px

# 1. Generate some mock data (e.g., Engine RPM vs. Fuel Consumption in an embedded system)
np.random.seed(42)
rpm = np.random.uniform(1000, 6000, 100)
fuel_cons = (rpm * 0.005) + np.random.normal(0, 5, 100) + 10
load_status = np.random.choice(["Low Load", "Medium Load", "High Load"], 100)

df = pd.DataFrame(
    {"RPM": rpm, "FuelConsumption": fuel_cons, "EngineLoad": load_status}
)

# 2. Create an interactive scatter plot
fig = px.scatter(
    df,
    x="RPM",
    y="FuelConsumption",
    color="EngineLoad",  # Automatic color grouping
    trendline="ols",  # Adds an Ordinary Least Squares linear regression line
    title="Engine Performance: Fuel Consumption vs. RPM",
    labels={
        "RPM": "Engine Speed (RPM)",
        "FuelConsumption": "Fuel Consumption (L/h)",
    },
    hover_data=["EngineLoad"],  # Extra data to display on hover
    template="plotly_dark",  # Clean, dark sleek UI template
)

# 3. Render the plot in your browser or Jupyter notebook
fig.show()

# 4. (Optional) Save it as a standalone interactive HTML file
# fig.write_html("engine_report.html")
