import random as rd

from utils.magic import magicSeed

def getRandomColor(num):
    rd.seed(magicSeed())
    choosed = rd.sample(colors, num)

    return choosed

colors = ['black', #1
          'darkgray',  #2
          'gainsboro',  #3
          'rosybrown',  #4
          'lightcoral',  #5
          "indianred",  #6
          'firebrick',  #7
          'maroon',  #8
          'red',  #9
          'coral',  #10
          'orangered',  #11
          'sienna',  #12
          'chocolate',  #13
          'sandybrown',  #14
          'bisque',  #15
          'wheat',  #16
          'gold',  #17
          'khaki',  #18
          'olive',  #19
          'lightyellow',  #20
          'yellow',  #21
          'olivedrab',  #22
          'yellowgreen',  #23
          'darkseagreen',  #24
          'palegreen',  #25
          'darkgreen',  #26
          'lime',  #27
          'mediumseagreen',  #28
          'aquamarine',  #29
          'turquoise',  #30
          'lightseagreen',  #31
          'lightcyan',  #32
          'teal',  #33
          'lightblue',  #34
          'deepskyblue',  #35
          'steelblue',  #36
          'navy',  #37
          'blue',  #38
          'slateblue',  #39
          'darkslateblue',  #40
          'mediumpurple',  #41
          'blueviolet',  #42
          'plum',  #43
          'violet',  #44
          'purple',  #45
          'magenta',  #46
          'mediumvioletred',  #47
          'hotpink',  #48
          'crimson',  #49
          'pink',  #50
          ]


