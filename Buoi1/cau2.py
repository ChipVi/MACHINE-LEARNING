scores=[]
for i in range(10):
    a = int(input('Enter a score '+ str(i+1) +': '))
    scores.append(a)

for score in scores:
    if (score>100) :
        print('invalid score')

scores.sort()

print('1a. Lowest score is ', scores[0])
print('1b. Highest score is ', scores[9])
print('2. Average score is', sum(scores)/len(scores))
print('3. The second highest score is', scores[8])

caud= scores[2:]
print('4. The average score after dropping 2 lowest scores is', sum(caud)/len(caud))