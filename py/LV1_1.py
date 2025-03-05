print('Unesi sate')
workHours = int(input())

print('Unesi satnicu')
payPerHour = int(input())


def total_euro():
    pay=workHours*payPerHour
    print('Zarada je:',pay,'€')
    
total_euro()