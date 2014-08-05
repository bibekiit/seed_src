import numpy as np
import copy




def tico(img, x, y, oimg, orient_img, o_rel, radius, theta_count, c_flag, start_t, blk_sz):
    """img = 2D array; x, y = scalar; orient_img, o_rel = 2D array; radius, theta_count = 1Darray;
    c_flag = boolean variable; start_t = scalar. oimg and blk_sz are not being used (consider removing)"""
   

    Xt = 0
    Yt = 0

    orients = []
    radii = []
    index = 1

    minX = 0
    minY = 0
    minT = 2*(np.pi)

    start_theta = 0

    if c_flag == 1:
        start_theta = start_t
    else:
        start_theta = orient_img[y - 1, x - 1]

    theta = start_theta
    tmp_radius = copy.deepcopy(radius)
    o_index = 1
    r_index = 1
    o_i = 0

    while r_index <= radius.size:

        while (o_i < theta_count[o_index - 1]) and not c_flag or (c_flag and o_i < theta_count[o_index - 1]/2.0):

            a = x + radius(r_index - 1)*np.cos(theta)
            b = y - radius(r_index - 1)*np.sin(theta)

            Xt = int(np.ceil(a) if a >= 0 else np.floor(a))
            Yt = int(np.ceil(b) if b >= 0 else np.floor(b))

            good = 1

            if Xt <= 20 or Xt >= img.shape[1] - 20 or Yt <= 20 or Yt >= img.shape[0] - 20:
                good = 0

            if good == 0 or ~np.isnan(o_rel[Yt - 1, Xt - 1]) and o_rel[Yt - 1, Xt - 1] < 0.5:
                radii[index - 1] = -1
                orients[index - 1] = -1
            else:
                t_a = []
                c = 0
                for e in range(Yt - 1, Yt + 2):
                    for f in range(Xt - 1, Xt + 2):
                        c = c + 1
                        t_a[c - 1] = orient_img[e - 1, f - 1]

                t_a = np.median(t_a)
                t_b = start_theta

                radii[index - 1] = radius[r_index - 1]
                orients[index - 1] = min(abs(t_b - t_a), abs(np.pi - abs(t_b - t_a)))*2/np.pi

                if orients[index - 1] < minT:
                    minT = orients[index - 1]
                    minX = Xt
                    minY = Yt
                    
            o_i = o_i + 1
            index = index + 1
            theta = theta + o_i*2*np.pi/theta_count[o_index - 1]

        r_index = r_index + 1
        theta = start_theta
        o_index = o_index + 1
        o_i = 0

    return orients, radii, minX, minY, minT

            
    
