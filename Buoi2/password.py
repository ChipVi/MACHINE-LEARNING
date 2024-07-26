passw = 'hiih'
for i in range(5):
    if input('enter password: ') == passw:
        print('welcome')
        break
    elif i == 4:
        print('you have entered wrong password 5 times')
        break
    else:
        print('try again')
