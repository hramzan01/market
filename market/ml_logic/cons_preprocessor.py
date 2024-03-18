import numpy as np
import string

def house_select(X):
    '''
     A function which takes into account a list of three user inputs on the house
     X = [House price, Income, House type]

     returns the ACORN categorisation to predict household energy consumption
    '''

    # define categories weightings from ACORN London dataset
    # House Price
    # A = '<100k', B = '100k-150k',  = '150k-250k'
    # D = '250k-500k', E = '500k-750k', F = '750k-1m', G = '>1000k'
    Ha = [13,17,16,38,40,35,36,35,36,37,144,180,159,205,252,226,326]
    Hb = [35,32,46,40,85,89,77,133,163,112,152,149,142,204,123,78,103]
    Hc = [42,66,117,39,83,144,154,143,145,193,91,90,85,65,59,58,62]
    Hd = [76,161,146,123,136,119,107,88,69,62,82,65,86,41,59,98,24]
    He = [111,268,164,174,184,68,99,92,90,52,26,25,24,25,46,57,48]
    Hf = [156,317,201,89,94,41,44,41,42,43,49,48,46,48,84,82,89]
    Hg = [1805,124,56,1002,111,43,47,43,44,45,31,30,29,30,36,35,38]
    # Based on user inputs
    if X[0] < 100000:
        H = 'a'
    elif X[0] < 150000:
        H = 'b'
    elif X[0] < 250000:
        H = 'c'
    elif X[0] < 500000:
        H = 'd'
    elif X[0] < 750000:
        H = 'e'
    elif X[0] < 1000000:
        H = 'f'
    else:
        H = 'g'

    # Income
    Ia = [19,30,65,43,55,78,60,73,131,70,115,111,129,219,134,154,218]
    Ib = [49,69,101,79,95,111,101,109,114,108,118,120,117,76,119,112,82]
    Ic = [112,127,123,125,131,119,130,124,80,128,94,97,84,33,79,64,28]
    Id = [200,189,134,170,152,113,142,121,58,125,66,69,55,14,47,35,10]
    Ie = [304,247,138,209,160,100,140,108,42,110,44,45,34,6,27,20,4]
    If = [11,332,131,267,152,74,119,80,24,78,24,22,16,2,11,9,1]
    # Based on user inputs
    if X[1] < 20000:
        I = 'a'
    elif X[1] < 40000:
        I = 'b'
    elif X[1] < 60000:
        I = 'c'
    elif X[1] < 80000:
        I = 'd'
    elif X[1] < 100000:
        I = 'e'
    else:
        I = 'f'

    # House Type
    #a = Bungalow, b = Detached house, c =Flat or maisonette
    #d=Semi-detached house, e = Terraced house
    ta = [118,100,198,51,51,167,94,74,410,55,61,64,64,64,45,50,50]
    tb = [431,419,229,46,98,287,203,59,138,61,38,32,25,24,20,19,13]
    tc = [36,25,47,274,196,17,22,17,54,30,299,49,29,159,160,142,277]
    td = [56,70,139,36,66,119,96,192,172,102,33,72,139,125,39,92,52]
    te = [23,20,25,86,77,38,103,71,31,163,77,199,140,72,169,123,83]

    X[2] = X[2].lower()
    if X[2] == 'bungalow':
        t = 'a'
    elif X[2] == 'detached':
        t = 'b'
    elif X[2] == 'flat' or X[2] == 'maisonette':
        t = 'c'
    elif X[2] == 'semi-detached':
        t = 'd'
    else:
        t = 'e'

    # define weightings based on house price
    if H == 'a':
        H_array = np.array(Ha)
        total = np.sum(H_array)
        weights_H = H_array/total
    elif H == 'b':
        H_array = np.array(Hb)
        total = np.sum(H_array)
        weights_H = H_array/total
    elif H == 'c':
        H_array = np.array(Hc)
        total = np.sum(H_array)
        weights_H = H_array/total
    elif H == 'd':
        H_array = np.array(Hd)
        total = np.sum(H_array)
        weights_H = H_array/total
    elif H == 'e':
        H_array = np.array(He)
        total = np.sum(H_array)
        weights_H = H_array/total
    elif H == 'f':
        H_array = np.array(Hf)
        total = np.sum(H_array)
        weights_H = H_array/total
    else:
        H_array = np.array(Hg)
        total = np.sum(H_array)
        weights_H = H_array/total

    # define weightings based on income
    if I == 'a':
        I_array = np.array(Ia)
        total = np.sum(I_array)
        weights_I = I_array/total
    elif I == 'b':
        I_array = np.array(Ib)
        total = np.sum(I_array)
        weights_I = I_array/total
    elif I == 'c':
        I_array = np.array(Ic)
        total = np.sum(I_array)
        weights_I = I_array/total
    elif I == 'd':
        I_array = np.array(Id)
        total = np.sum(I_array)
        weights_I = I_array/total
    elif I == 'e':
        I_array = np.array(Ie)
        total = np.sum(I_array)
        weights_I = I_array/total
    elif I == 'f':
        I_array = np.array(If)
        total = np.sum(I_array)
        weights_I = I_array/total
    else:
        I_array = np.array(Ia)
        total = np.sum(I_array)
        weights_I = I_array/total

    # define weightings based on house type
    if t == 'a':
        t_array = np.array(ta)
        total = np.sum(t_array)
        weights_t = t_array/total
    elif t == 'b':
        t_array = np.array(tb)
        total = np.sum(t_array)
        weights_t = t_array/total
    elif t == 'c':
        t_array = np.array(tc)
        total = np.sum(t_array)
        weights_t = t_array/total
    elif t == 'd':
        t_array = np.array(td)
        total = np.sum(t_array)
        weights_t = t_array/total
    else:
        t_array = np.array(te)
        total = np.sum(t_array)
        weights_t = t_array/total

    # combine weights to find weighting for each acorn category
    total_weight = weights_H + weights_I + weights_t
    total_weight = list(total_weight)
    total_weight

    # find most likely acorn category
    max_val = max(total_weight)
    idx_max = total_weight.index(max_val)
    idx_max

    # print most likely acorn
    ACORN = string.ascii_uppercase[idx_max]

    return ACORN

house_type = house_select([743000, 100000, 'flat'])
print(house_type)
