from time import time

magic = 7355605

def magicSeed():
    return time()%magic

def magicList():
    return [i for i in range(magic)]