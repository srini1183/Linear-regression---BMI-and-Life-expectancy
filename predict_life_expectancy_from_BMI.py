# Importing libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Loading the data
dataframe = pd.read_csv('data/bmi_and_life_expectancy.csv')
x = dataframe[['BMI']]
y = dataframe[['Life expectancy']]

# Split data for training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y)


model = LinearRegression()
model.fit(x_train,y_train)

# Visualize
plt.scatter(x_test, y_test)
plt.plot(x_test, model.predict(x_test))
plt.show()


