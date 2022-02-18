'''
Random Password Generator using Python
Author: Ayushi Rawat
'''

#import the necessary modules!
import random
import string

#   print('hello, Welcome to Password generator!')

#input the length of password
#   length = int(input('\nEnter the length of password: '))
i = 0

punct = ["@", "*", "#", "?", "!", "+", "&"]
x = 0
#define data7
while i < 4:
    i = i+1
    lower = string.ascii_lowercase
    upper = string.ascii_uppercase
    num = string.digits
    # symbols = string.punctuation
    symbols = random.choice(punct)

#string.ascii_letters

#combine the data
    all = lower + symbols + upper + num

#use random
    temp = random.sample(all,15)

#create the password
    password = "".join(temp)

#print the password
    print(password)

