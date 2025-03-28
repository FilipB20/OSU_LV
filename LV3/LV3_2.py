import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')
plt.figure()
data['CO2 Emissions (g/km)'].plot(kind='hist', bins = 20, figsize=(10,6))
plt.title('CO2 Emissions (g/km)')
plt.show()

fuels = {'X' : 'blue', 'Z' : 'pink', 'D' : 'red', 'E' : 'yellow', 'N' : 'green'}
plt.figure(figsize=(10,6))
for fuel_type, color in fuels.items():
    subset = data[data['Fuel Type'] == fuel_type]
    plt.scatter(subset['Fuel Consumption City (L/100km)'], subset['CO2 Emissions (g/km)'],
                color=color, label=fuel_type)
plt.legend(title='Fuel Type')
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.show()

groupedby_fuel_type = data.groupby('Fuel Type')
groupedby_fuel_type.boxplot(column=['Fuel Consumption Hwy (L/100km)'], sharey=False, figsize=(10,6))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.5, hspace=0.5)
plt.show()

vehicle_count = groupedby_fuel_type.size()
vehicle_count.plot(kind='bar', color='skyblue', figsize=(10,6))
plt.title('Broj vozila po tipu goriva')
plt.xlabel('Tip goriva')
plt.ylabel('Broj vozila')
plt.show()

groupedby_cylinders = data.groupby('Cylinders')
plt.figure(figsize=(10,6))
plt.bar(groupedby_cylinders.size().keys(), groupedby_cylinders['CO2 Emissions (g/km)'].mean())
plt.title('Prosječna CO2 emisija po broju cilindara')
plt.xlabel('Broj cilindara')
plt.ylabel('Prosječna CO2 emisija')
plt.show()