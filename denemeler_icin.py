import random
import nn_params

'''
def rand_key(length):
    random_bin = ""
    for i in range(length):
        random_bin += str(random.randint(0, 1))
    return(random_bin)


random_bin = rand_key(4)
i = 0
while(i == 0):
    for key,value in nn_params.activations().items():
        if value == random_bin:
            i += 1
            print(key)
    random_bin = rand_key(4)
'''

def random_encode():
    result = ""

    activations = nn_params.activations()
    a = random.randrange(len(activations))
    result += str(activations[a])
    result += f'{a:04b}'

    optimizers = nn_params.optimizers()
    b = random.randrange(len(optimizers))
    result += str(optimizers[b])
    result += f'{b:04b}'

    return result

def decode():
    return None

print(random_encode())