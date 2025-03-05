words = {}
 
try:
    with open("song.txt") as file:
 
        for row in file:
 
            row = row.lower()
            for sign in ",.?!;:":
                row = row.replace(sign, "")
 
            list_words = row.split()
 
            for word in list_words:
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1
 
    words_once = [word for word, broj in words.items() if broj == 1]
 
    print("Ukupan broj različitih riječi:", len(words))
    print("Broj riječi koje se pojavljuju samo jednom:", len(words_once))
    print("Riječi koje se pojavljuju samo jednom:")
    print(", ".join(words_once))
 
except FileNotFoundError:
    print("Greška: Datoteka 'song.txt' nije pronađena.")