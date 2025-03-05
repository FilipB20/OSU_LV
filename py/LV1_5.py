spam = 0
spamwords = 0
ham = 0
endwith = 0
hamwords = 0

fhand = open("SMSSpamCollection.txt")
for line in fhand :

    line = line.rstrip()
    words = line.split("\t")
    wordsline = words[1].split()
    if words[0] == "spam":
        spam += 1
        spamwords += len(wordsline)
        if words[1].endswith('!'):
            endwith += 1
    if words[0] == "ham":
        ham += 1
        hamwords += len(wordsline)

    
    
fhand.close()
avgSpamWords = float(spamwords)/float(spam)
avgHamWords = float(hamwords)/float(ham)
print(f"Prosječan broj riječi u spam porukama {avgSpamWords}")
print(f"Prosječan broj riječi u ham porukama {avgHamWords}")
print(f"Broj spam poruka koji završavaju uskličnikom: {endwith}")