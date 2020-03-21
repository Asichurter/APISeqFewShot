

def statParamNumber(model):
    num_of_params = 0
    for par in model.parameters():
        num_of_params += par.numel()
    print('params:', num_of_params)