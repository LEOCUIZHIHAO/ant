"""
@Authors Leo.cui
22/5/2018
One_hot_encoder functions

"""

def one_hot_encode(label_set):

    label = []
    white = [1,0]
    black = [0,1]

    #one-hot label
    for i in label_set:
        if i == 0:
            label.append(white)
        elif i ==1:
            label.append(black)
        elif i ==-1:
          print("label -1")

    return label
