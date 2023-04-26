import numpy as np
import pandas as pd
import statsmodels.api as sm

# Define input and output variables
X = cities[['PD', 'LT', 'AQI', 'GDP', 'UR']].astype('float')
y = cities['Level'].astype('float')

# Fit linear regression model
model = sm.OLS(y, X).fit()

# Print model summary
print(model.summary())

# Estimate impact of each input variable on the score
for col in X.columns:
    new_X = X.copy()
    new_X[col] = X[col].mean()

    predicted_score = model.predict(new_X)
    impact_on_score = predicted_score.mean() - y.mean()

    print(f"The impact of increasing {col} by one unit is {impact_on_score:.2f}")