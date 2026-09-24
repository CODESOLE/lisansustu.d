import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy as sci

# 1. Generate some mock data (e.g., Engine RPM vs. Fuel Consumption in an embedded system)
np.random.seed(42)
rpm = np.random.uniform(1000, 6000, 100)
fuel_cons = (rpm * 0.005) + np.random.normal(0, 5, 100) + 10
load_status = np.random.choice(["Low Load", "Medium Load", "High Load"], 100)

def softmax(data):
    expp = np.exp(data)
    sum_expp = np.sum(expp)
    out = [x / sum_expp for x in expp]
    for d  in out:
        print(d)
    return out
l = softmax([1, 3, 5, 3, 1])
print('sum of softmax elemets: ', np.sum(l)) # indeed equals to 1

df = pd.DataFrame(
    {"RPM": rpm, "FuelConsumption": fuel_cons, "EngineLoad": load_status}
)

# Create the scatter plot
sns.scatterplot(data=df, x="RPM", y="FuelConsumption")
sns.boxplot()

# Customize canvas elements explicitly
plt.title("EngineLoad vs. FuelConsumption")
plt.xlabel("EngineLoad")
plt.ylabel("FuelConsumption")
plt.show()
