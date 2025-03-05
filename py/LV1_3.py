numberList = []
print("Unesi broj, kad napises Done program zavrsava")

while True:
    inputAnswer=input()
    if inputAnswer.lower() == "done":
        break
    elif inputAnswer != str:
        numberList.append(int(inputAnswer))

cntr = len(numberList)
print (numberList)
print("Ukupan broj elemanata: ", cntr)
print ("Najmanji el: ", min(numberList))
print ("Najveci el: ", max(numberList))

sum = sum(numberList)

average = sum / cntr

print(average)

numberList.sort()
print("Sortirana list: \n", numberList)