import pandas as pd

data = pd.read_csv('data_C02_emission.csv')
print(f"DataFrame sadrži {len(data)} mjerenja.")
print(f"Tipovi veličina: \n{data.dtypes}")
print(data.isnull().sum())
data.drop_duplicates()
columns_to_convert = ['Make','Model','Vehicle Class','Transmission','Fuel Type']
data[columns_to_convert] = data[columns_to_convert].astype('category')
print(data.dtypes)

data_sorted = data.sort_values(by=['Fuel Consumption City (L/100km)'])
print(f"Tri auta s najvećom gradskom potrošnjom: \n{data_sorted[['Make','Model','Fuel Consumption City (L/100km)']].tail(3)}")
print(f"Tri auta s najmanjom gradskom potrošnjom: \n{data_sorted[['Make','Model','Fuel Consumption City (L/100km)']].head(3)}")

filtered_vehicles = data[(data['Engine Size (L)'] > 2.5) & (data['Engine Size (L)'] < 3.5)]
print(f"{filtered_vehicles.shape[0]} vozila ima veličinu motora između 2.5 i 3.5 L.")
print(f"Prosječna CO2 emisija plinova za ova vozila je {filtered_vehicles['CO2 Emissions (g/km)'].mean().__round__(2)}")

audi = data[data['Make'] == 'Audi']
print(f"{audi.shape[0]} mjerenja se odnosi na vozila proizvođača Audi.")
audi_4_cylinders = audi[audi['Cylinders'] == 4]
print(f"Prosječna emisija CO2 plinova automobila proizvođača Audi koji imaju 4 cilindra je {audi_4_cylinders['CO2 Emissions (g/km)'].mean().__round__(2)}")

data_groupedby_cylinders = data.groupby('Cylinders')
data_groupedby_even_cylinders = data[data['Cylinders'] % 2 == 0].groupby('Cylinders')
print(f"Broj vozila prema broju cilindara: \n{data_groupedby_cylinders.size()}")
print(f"Broj vozila prema broju cilindara (samo parni): \n{data_groupedby_even_cylinders.size()}")
print(f"Prosječna emisija CO2 plinova prema broju cilindara: \n{data_groupedby_cylinders['CO2 Emissions (g/km)'].mean().__round__(2)}")

diesel_vehicles = data[data['Fuel Type'] == 'D']
regular_gasoline_vehicles = data[data['Fuel Type'] == 'X']
print(f"Prosječna gradska potrošnja za dizel vozila je {diesel_vehicles['Fuel Consumption City (L/100km)'].mean().__round__(2)}, a za regularni benzin {regular_gasoline_vehicles['Fuel Consumption City (L/100km)'].mean().__round__(2)}")
print(f"Medijalne vrijednosti: Dizel - {diesel_vehicles['Fuel Consumption City (L/100km)'].median().__round__(2)}, Benzin - {regular_gasoline_vehicles['Fuel Consumption City (L/100km)'].median().__round__(2)}")

print(f"Dizel vozilo s 4 cilindra s najvećom gradskom potrošnjom: \n{diesel_vehicles[diesel_vehicles['Cylinders'] == 4].sort_values(by='Fuel Consumption City (L/100km)').tail(1)}")

manual_transmission_vehicles = data[data['Transmission'].apply(lambda x: x.startswith('M'))]
print(f"{manual_transmission_vehicles.shape[0]} vozila ima ručni tip mjenjača.")

print(data.corr(numeric_only=True))
