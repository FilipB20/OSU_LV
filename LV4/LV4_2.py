import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import sklearn.linear_model as lm

data = pd.read_csv('data_C02_emission.csv')
ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(data[['Fuel Type']]).toarray()

numerical_features = data.select_dtypes(include='number')
ohe_columns = ohe.get_feature_names_out(['Fuel Type'])
X_encoded_df = pd.DataFrame(X_encoded, columns=ohe_columns, index=data.index)
numerical_features = pd.concat([numerical_features, X_encoded_df], axis=1)

X_train, X_test, y_train, y_test = train_test_split(numerical_features.drop(['CO2 Emissions (g/km)'], axis=1), 
                                                    numerical_features['CO2 Emissions (g/km)'], test_size=0.2, random_state=1)

linearModel = lm.LinearRegression()
linearModel.fit(X_train, y_train)
print("Model coefficients:")
print(linearModel.coef_)


y_test_prediction = linearModel.predict(X_test)
plt.scatter(y_test, y_test_prediction)
plt.title('Real Values vs. Predicted values')
plt.xlabel("Real Values")
plt.ylabel("Predicted Values")
plt.grid(True)
plt.show()

absolute_errors = abs(y_test - y_test_prediction)

max_error_index = absolute_errors.idxmax()
max_error = absolute_errors[max_error_index]

vehicle_model = data.loc[max_error_index, 'Model']

print(f"Maximum absolute error: {max_error}")
print(f"Model of the vehicle associated with maximum error: {vehicle_model}")