import cvxpy as cp
import numpy as np
from tqdm import tqdm
from collections import deque

R_lower = 20
R_upper = 20
R_bottom = 50


def get_gain(x):
    return np.sum((np.abs(x))**2)


def get_h(n=1):

    x = np.random.normal(0, np.sqrt(0.5), size=(n)) +\
        1j*np.random.normal(0, np.sqrt(0.5), size=(n))

    return x


def in_db(x):
    temp = x/10
    temp = 10**temp
    return temp


def get_SNR_main(a, b):
    temp = (a*b)/((a+1)**2)
    return b/(temp+2)


def get_SNR_eve(a, c, d):
    temp1 = d*a*(a+c+1)
    temp2 = ((a+1)**2)*(c+1)
    return d/((temp1/temp2)+2)


def get_rate_lower(hAB, hBA, SAB, SBA, alpha, beta, nE):

    SEA = alpha*SBA
    SEB = beta*SAB

    a = SBA*get_gain(hBA)
    b = SAB*get_gain(hAB)

    data = np.zeros(R_lower)

    for i in range(R_lower):

        gA = get_h(nE)
        gB = get_h(nE)

        c = SEA*get_gain(gA)
        d = SEB*get_gain(gB)

        temp1 = np.log2(1+get_SNR_main(a, b)) - \
            np.log2(1+get_SNR_eve(a, c, d))
        temp1 = max(0, temp1)

        data[i] = temp1

    return np.mean(data)


def get_rate_lower_classic(hAB, hBA, SAB, SBA, alpha, beta, nE):

    SEA = alpha*SBA
    SEB = beta*SAB

    a = SBA*get_gain(hBA)
    b = SAB*get_gain(hAB)

    data = np.zeros(R_lower)

    for i in range(R_lower):

        gA = get_h(nE)
        gB = get_h(nE)

        c = SEA*get_gain(gA)
        d = SEB*get_gain(gB)

        temp1a = np.log2(1+a)-np.log2(1+c)
        temp1b = max(0, temp1a)

        temp2a = np.log2(1+b)-np.log2(1+d)
        temp2b = max(0, temp2a)

        temp3 = temp1b+temp2b

        data[i] = temp3

    return np.mean(data)


def get_rate_middle(hABs, hBAs, SABs, SBAs, alpha, beta, nE):

    sum_rate = 0

    for i in range(len(hABs)):

        hAB = hABs[i]
        hBA = hBAs[i]
        SAB = SABs[i]
        SBA = SBAs[i]

        rate = get_rate_lower(hAB, hBA, SAB, SBA, alpha, beta, nE)
        sum_rate += rate

    return sum_rate/len(hABs)


def get_rate_middle_classic(hABs, hBAs, SABs, SBAs, alpha, beta, nE):

    sum_rate = 0

    for i in range(len(hABs)):

        hAB = hABs[i]
        hBA = hBAs[i]
        SAB = SABs[i]
        SBA = SBAs[i]

        rate = get_rate_lower_classic(hAB, hBA, SAB, SBA, alpha, beta, nE)
        sum_rate += rate

    return sum_rate/len(hABs)


def get_rate_upper_nopower(SAB_avg, SBA_avg, alpha, beta, N, nE):

    SABs = [SAB_avg]*N
    SBAs = [SBA_avg]*N

    policy_1 = np.zeros(R_upper)
    policy_2 = np.zeros(R_upper)
    policy_3 = np.zeros(R_upper)
    count = 0

    for i in tqdm(range(R_upper)):

        hABs = get_h(N)
        hBAs = get_h(N)

        policy_1[i] = get_rate_middle(hABs, hBAs, SABs, SBAs, alpha, beta, nE)

        hABs, hBAs = pair_indices_by_value(hABs, hBAs, same_direc=True)
        policy_2[i] = get_rate_middle(hABs, hBAs, SABs, SBAs, alpha, beta, nE)

        hABs, hBAs = pair_indices_by_value(hABs, hBAs, same_direc=False)
        policy_3[i] = get_rate_middle(hABs, hBAs, SABs, SBAs, alpha, beta, nE)
        
        if policy_1[i]>policy_2[i] or policy_3[i]>policy_2[i]:
            print('problem')
            count+=1
    
    print(count/R_upper)

    return np.mean(policy_1), np.mean(policy_2), np.mean(policy_3)


def get_rate_upper_both(SAB_avg, SBA_avg, alpha, beta, N, nE):

    policy_2 = np.zeros(R_upper)
    policy_4 = np.zeros(R_upper)
    policy_5 = np.zeros(R_upper)
    countB = 0

    for i in tqdm(range(R_upper)):
        while True:
            try:
                SABs = [SAB_avg]*N
                SBAs = [SBA_avg]*N

                hABs = get_h(N)
                hBAs = get_h(N)

                hABs, hBAs = pair_indices_by_value(hABs, hBAs, same_direc=True)
                policy_2[i] = get_rate_middle(hABs, hBAs, SABs, SBAs,
                                              alpha, beta, nE)

                SABs_new = power_alloc_Bob(hABs, hBAs, SABs, SBAs,
                                           alpha, beta, nE, updt=False)
                policy_4[i] = get_rate_middle(hABs, hBAs, SABs_new, SBAs,
                                              alpha, beta, nE)

                SBAs_new = power_alloc_Alice(SABs_new, SBAs)
                policy_5[i] = get_rate_middle(hABs, hBAs, SABs_new, SBAs_new,
                                              alpha, beta, nE)
                break
            except Exception as e:
                countB += 1
                print(f"An error occurred: {e}")
    print(countB)
    return np.mean(policy_2), np.mean(policy_4), np.mean(policy_5)


def get_rate_upper_fin(SAB_avg, SBA_avg, alpha, beta, N, nE):

    policy_5 = np.zeros(R_upper)
    policy_6 = np.zeros(R_upper)
    countB = 0

    for i in tqdm(range(R_upper)):
        while True:
            try:
                SABs = [SAB_avg]*N
                SBAs = [SBA_avg]*N

                hABs = get_h(N)
                hBAs = get_h(N)

                hABs, hBAs = pair_indices_by_value(hABs, hBAs, same_direc=True)
                SABs_new = power_alloc_Bob(hABs, hBAs, SABs, SBAs,
                                           alpha, beta, nE, updt=False)
                SBAs_new = power_alloc_Alice(SABs_new, SBAs)
                policy_5[i] = get_rate_middle(hABs, hBAs, SABs_new, SBAs_new,
                                              alpha, beta, nE)
                
                for jj in range(100):
                    print(jj)
                    SABs_old = SABs_new
                    SBAs_old = SBAs_new
                    SABs_new = power_alloc_Bob(hABs, hBAs, SABs, SBAs_old,
                                               alpha, beta, nE, updt=False)
                    SBAs_new = power_alloc_Alice(SABs_new, SBAs)
                    print(SABs_new,SABs_old)
                    if np.linalg.norm(np.array(SABs_new)-
                                      np.array(SABs_old))<1e-1:
                        # and
                        # np.linalg.norm(np.array(SBAs_new)-
                        #                np.array(SBAs_old))<1e-1:
                        break
                
                break
            
                policy_6[i] = get_rate_middle(hABs, hBAs, SABs_new, SBAs_new,
                                              alpha, beta, nE)
            
            except Exception as e:
                countB += 1
                print(f"An error occurred: {e}")
    print(countB)
    return np.mean(policy_5), np.mean(policy_6)


def get_rate_fin(SAB_avg, SBA_avg, alpha, beta, N, nE):

    policy_5 = np.zeros(R_upper)
    policy_0 = np.zeros(R_upper)
    countB = 0

    for i in tqdm(range(R_upper)):
        while True:
            try:
                SABs = [SAB_avg]*N
                SBAs = [SBA_avg]*N

                hABs = get_h(N)
                hBAs = get_h(N)

                SABs_new, SBAs_new =\
                    power_alloc_classic(hABs, hBAs, SABs, SBAs)
                # print(sum(SABs_new),sum(SABs))
                # print(sum(SBAs_new),sum(SBAs))

                policy_0[i] = get_rate_middle_classic(hABs, hBAs,
                                                      SABs, SBAs,
                                                      alpha, beta, nE)

                hABs, hBAs = pair_indices_by_value(hABs, hBAs, same_direc=True)
                SABs_new = power_alloc_Bob(hABs, hBAs, SABs, SBAs,
                                           alpha, beta, nE, updt=False)
                SBAs_new = power_alloc_Alice(SABs_new, SBAs)
                policy_5[i] = get_rate_middle(hABs, hBAs, SABs_new, SBAs_new,
                                              alpha, beta, nE)

                break
            except Exception as e:
                countB += 1
                print(f"An error occurred: {e}")
    print(countB)
    return np.mean(policy_5), np.mean(policy_0)


def pair_indices_by_value(arr1, arr2, same_direc):

    arr1_squared = np.abs(arr1)**2
    arr2_squared = np.abs(arr2)**2

    sorted_indices_arr1 = np.argsort(arr1_squared)[::-1]
    if same_direc:
        sorted_indices_arr2 = np.argsort(arr2_squared)[::-1]
    else:
        sorted_indices_arr2 = np.argsort(arr2_squared)

    return arr1[sorted_indices_arr1], arr2[sorted_indices_arr2]


def function_g(hAB, hBA, SAB, SBA):
    an = SBA*((np.abs(hBA))**2)
    qn = (np.abs(hAB))**2
    rn = (an*qn)/((an+1)**2)
    temp1 = (qn*SAB)/((rn*SAB)+2)
    temp2 = ((rn*SAB)+2)**2
    temp3 = np.log2(np.e)/(1+temp1)
    temp4 = (2*qn)/temp2
    return temp3*temp4


def function_v(hAB, hBA, SAB, SBA):
    an = SBA*get_gain(hBA)
    qn = get_gain(hAB)
    rn = (an*qn)/((an+1)**2)
    temp1 = (qn*SAB)/((rn*SAB)+2)
    return np.log2(1+temp1)


def get_ln(hAB, hBA, SAB, SBA, alpha, beta, nE):

    an = SBA*get_gain(hBA)

    SEA = alpha*SBA
    SEB = beta*SAB

    data = np.zeros(R_bottom)
    for i in range(R_bottom):

        gA = get_h(nE)
        gB = get_h(nE)

        c = SEA*get_gain(gA)
        d = SEB*get_gain(gB)

        temp1 = np.log2(1+get_SNR_eve(an, c, d))

        data[i] = temp1

    return np.mean(data)


def find_SAB_for_mu(mu, hAB, hBA, SBA, p_upper):

    tol = 1e-4
    max_iter = 20

    p_lower = 0
    p_guess = (p_lower + p_upper)*0.5

    iteration = 0
    while iteration < max_iter:

        g_p_guess = function_g(hAB, hBA, p_guess, SBA)

        if abs(g_p_guess - mu) < tol:
            # print(iteration)
            return p_guess
        if g_p_guess < mu:
            p_upper = p_guess
        else:
            p_lower = p_guess

        p_guess = (p_lower + p_upper) / 2.0
        iteration += 1

    return p_guess


def power_alloc_Bob(hABs, hBAs, SABs, SBAs, alpha, beta, nE, updt=False):

    p_total = np.sum(SABs)

    mu = 1e-6
    mu1 = 0
    mu2 = None
    thres = 0
    p_thres = 1e-1
    max_iter = 50

    II = [i for i in range(len(SABs))]
    SABns = np.zeros(len(II))
    fifo_queue = deque([0]*5)

    if not updt:
        lns = np.zeros(len(II))
        for i in II:
            lns[i] = get_ln(hABs[i], hBAs[i], SABs[i], SBAs[i],
                            alpha, beta, nE)

    for count in range(max_iter):

        II_temp = []
        for i in II:

            SABn = find_SAB_for_mu(mu, hABs[i], hBAs[i], SBAs[i], p_total)
            vv = function_v(hABs[i], hBAs[i], SABn, SBAs[i])
            if updt:
                ln = get_ln(hABs[i], hBAs[i], SABn, SBAs[i],
                            alpha, beta, nE)
            else:
                ln = lns[i]
            vv = vv-ln
            if vv <= thres:
                SABns[i] = 0
                II_temp.append(i)
                mu1 = 0
                mu2 = None
            else:
                SABns[i] = SABn

        for i in II_temp:
            II.remove(i)

        if len(II) == 0:
            return SABns

        ssum = np.sum(SABns)
        if abs(ssum-p_total) < p_thres:
            # print(count)
            return SABns
        if ssum > p_total:
            mu1 = mu
            if mu2 == None:
                mu = mu*2
            else:
                mu = 0.5*(mu1+mu2)
        if ssum < p_total:
            mu2 = mu
            mu = 0.5*(mu1+mu2)

        fifo_queue.append(abs(p_total-ssum))
        _ = fifo_queue.popleft()
        if all_elements_same(fifo_queue):
            thres += 1e-2

    return None


def all_elements_same(queue):

    if not queue:
        return False

    first_element = queue[0]
    for element in queue:
        if element != first_element:
            return False

    return True


def power_alloc_Alice(SABs, SBAs):
    II = []
    for i, SAB in enumerate(SABs):
        if SAB > 0:
            II.append(i)
    p = np.sum(SBAs)/len(II)
    SBAs_new = np.zeros(len(SBAs))
    for j in II:
        SBAs_new[j] = p
    return SBAs_new


def water_filling(n, a, sum_x=1):

    x = cp.Variable(shape=n)
    alpha = cp.Parameter(n, nonneg=True)
    alpha.value = a

    obj = cp.Maximize(cp.sum(cp.log(alpha + x)))

    constraints = [x >= 0, cp.sum(x) - sum_x == 0]

    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.ECOS)
    if(prob.status == 'optimal'):
        return prob.status, prob.value, x.value
    else:
        return prob.status, np.nan, np.nan


def power_alloc_classic(hABs, hBAs, SABs, SBAs):
    hABs = np.abs(hABs)**2
    hBAs = np.abs(hBAs)**2
    # _, _, _, SABs_new, hABs_inv = GWF(sum(SABs), hABs, np.ones(len(hABs)))
    # _, _, _, SBAs_new, hBAs_inv = GWF(sum(SBAs), hBAs, np.ones(len(hBAs)))
    _, _, SABs_new = water_filling(len(hABs), 1/hABs, sum(SABs))
    _, _, SBAs_new = water_filling(len(hBAs), 1/hBAs, sum(SBAs))
    if abs(sum(SABs)-sum(SABs_new)) > 0.001:
        print('fail')
    if abs(sum(SBAs)-sum(SBAs_new)) > 0.001:
        print('fail')
    # print(hABs,SABs_new,hBAs,SBAs_new)
    return SABs_new, SBAs_new
