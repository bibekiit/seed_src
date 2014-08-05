import numpy as np
from p import p





def trace_path(img, m_list, path_len):
    """img, m_list are 2D arrays, path_len is a scalar. Output is a 2D array
    img_n with dimensions same as img."""

    img_n = np.zeros(img.shape)

    for index in range(m_list.shape[0]):
        
        x = m_list[index, 0]
        y = m_list[index, 1]
        CN = m_list[index, 2]
        
        progress = 1

        res = index
        img_n[y - 1, x - 1] = 1

        pax = 0
        pay = 0
        pbx = 0
        pby = 0
        pcx = 0
        pcy = 0

        pao = 0
        pbo = 0
        pco = 0             #1-8 position of 3x3 square around minutia

        for i in range(1,9):

            [ta, xa, ya] = p(img, x, y, i)
            [tb, xb, yb] = p(img, x, y, i + 1)

            if ta > tb:
                if pao == 0:
                    if i < 5:
                        pao = 4 + i
                    else:
                        pao = np.mod(4 + i, 9) + 1

                    pax = xa
                    pay = ya

                elif pbo == 0:
                    if i < 5:
                        pbo = 4 + i
                    else:
                        pbo = np.mod(4 + i, 9) + 1

                    pbx = xa
                    pby = ya

                else:
                    if i < 5:
                        pco = 4 + i
                    else:
                        pco = np.mod(4 + i, 9) + 1

                    pcx = xa
                    pcy = ya


        while progress < path_len:

            if pax == 0:
                 break

            img_n[pay - 1, pax - 1] = 1

            if pbx != 0:
                img_n[pby - 1, pbx - 1] = 1

            if pcx != 0:
                img_n[pcy - 1, pcx - 1] = 1

            progress = progress + 1

            if pao != 0:
                for i in range(1,9):
                    if i == pao:
                        continue

                    [ta, xa, ya] = p(img, pax, pay, i)

                    if ta == 1 and img_n[ya - 1, xa - 1] != 1:
                        if i < 5:
                            pao = 4 + i
                        else:
                            pao = np.mod(4 + i, 9) + 1

                        pax = xa
                        pay = ya
                        break

            if pbo != 0:
                for i in range(1, 9):
                    if i == pbo:
                        continue

                    [ta, xa, ya] = p(img, pbx, pby, i)

                    if ta == 1 and img_n[ya - 1, xa - 1] != 1:
                        if i < 5:
                            pbo = 4 + i
                        else:
                            pbo = np.mod(4 + i, 9) + 1

                        pbx = xa
                        pby = ya
                        break
                        

            if pco != 0:
                for i in range(1, 9):
                    if i == pco:
                        continue

                    [ta, xa, ya] = p(img, pcx, pcy, i)

                    if ta == 1 and img_n[ya - 1, xa - 1]  != 1:
                        if i < 5:
                            pco = 4 + i
                        else:
                            pco = np.mod(4 + i,9) + 1

                        pcx = xa
                        pcy = ya
                        break


    return img_n
                        

                    
                    
            

        
