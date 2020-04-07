from functools import reduce

def strlistToStr(l):
    s = reduce(lambda x,y: x+'/'+y, l)
    return s