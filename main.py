# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: For regression or time series analysis
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose

# Set plot style
sns.set(style="whitegrid")

# Load dataset (example: unemployment.csv)
# Make sure your CSV has columns like: Date, Region, Unemployment Rate
df = pd.read_csv("unemployment_10_years.csv")

# Preview data
print(df.head())

# Convert date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Basic info
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Plot Unemployment Rate over time (national level)
plt.figure(figsize=(12,6))
sns.lineplot(data=df, x="Date", y="Unemployment Rate", hue="Region")
plt.title("Unemployment Rate Over Time by Region")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Average Unemployment Rate by Region
plt.figure(figsize=(10,6))
avg_region = df.groupby('Region')['Unemployment Rate'].mean().sort_values()
sns.barplot(x=avg_region.values, y=avg_region.index, palette='coolwarm')
plt.title("Average Unemployment Rate by Region")
plt.xlabel("Average Rate (%)")
plt.ylabel("Region")
plt.tight_layout()
plt.show()

# Optional: Time series decomposition (for one region)
region_data = df[df['Region'] == 'India']  # replace 'India' with a valid region
region_data.set_index('Date', inplace=True)
result = seasonal_decompose(region_data['Unemployment Rate'], model='additive', period=12)
result.plot()
plt.suptitle('Time Series Decomposition - India')
plt.tight_layout()
plt.show()

# Optional: Linear Regression to check trend
df['Year'] = df['Date'].dt.year
grouped = df.groupby('Year')['Unemployment Rate'].mean().reset_index()

model = LinearRegression()
X = grouped[['Year']]
y = grouped['Unemployment Rate']
model.fit(X, y)

# Plot trend line
plt.figure(figsize=(8,5))
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, model.predict(X), color='red', label='Trend Line')
plt.title("Unemployment Trend Over Years")
plt.xlabel("Year")
plt.ylabel("Average Unemployment Rate (%)")
plt.legend()
plt.tight_layout()
plt.show()
