import pandas as pd
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

data = pd.read_csv('data_C02_emission.csv')

#a)
numerical_features = data.select_dtypes(include='number')
X_train, X_test, y_train, y_test = train_test_split(numerical_features.drop(['CO2 Emissions (g/km)'], axis=1),
                                                    numerical_features['CO2 Emissions (g/km)'], test_size=0.2, random_state=1)


#b)
plt.figure(figsize=(8, 6))

plt.scatter(X_train['Cylinders'], y_train, color='blue', label='Train Data', s=50)

plt.scatter(X_test['Cylinders'], y_test, color='red', label='Test Data', s=15)

plt.title('Ovisnost emisije CO2 plinova o broju cilindara')
plt.xlabel('Broj cilindara')
plt.ylabel('CO2 emisija')
plt.legend()
plt.grid(True)
plt.show()


#c)
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.hist(X_train['Cylinders'], bins=20)
plt.title('Histogram cilindara prije skaliranja')
plt.xlabel('Broj cilindara')
plt.ylabel('Broj vozila')

standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
scaled_X_train = minmax_scaler.fit_transform(X_train)
scaled_X_train_df = pd.DataFrame(scaled_X_train, columns=X_train.columns)

plt.subplot(1, 2, 2)
plt.hist(scaled_X_train_df['Cylinders'], bins=20, color='red')
plt.title('Histogram cilindara nakon skaliranja')
plt.xlabel('Skalirani broj cilindara')
plt.ylabel('Broj vozila')

plt.tight_layout()
plt.show()

scaled_X_test = minmax_scaler.transform(X_test)
scaled_X_test_df = pd.DataFrame(scaled_X_test, columns=X_test.columns)


#d)
linearModel = lm.LinearRegression()
linearModel.fit(scaled_X_train, y_train)
print("Koeficijenti modela:")
print(linearModel.coef_)


#e)
y_test_prediction = linearModel.predict(scaled_X_test)
plt.scatter(y_test, y_test_prediction)
plt.title('Stvarne vrijednosti vs. Predviđene vrijednosti')
plt.xlabel("Stvarne vrijednosti")
plt.ylabel("Predviđene vrijednosti")
plt.grid(True)
plt.show()


#f)
MAE = mean_absolute_error(y_test, y_test_prediction)
MAPE = mean_absolute_percentage_error(y_test, y_test_prediction)
MSE = mean_squared_error(y_test, y_test_prediction)
RMSE = math.sqrt(MSE)
R2 = r2_score(y_test, y_test_prediction)
print(f"MAE: {MAE}")
print(f"MAPE: {MAPE}")
print(f"MSE: {MSE}")
print(f"RMSE: {RMSE}")
print(f"R2: {R2}")


#g) Značajnija se razlika dobija tek kada ostane samo jedna ulazna veličina, a već s dvije ulazne veličine model dobro opisuje podatke