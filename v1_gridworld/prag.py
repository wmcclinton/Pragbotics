import numpy as np
from build_meaning_matrix import W, U, GRID
from env import render_world

MM = np.load("MeaningMatrix.npy")


# take intersection of hypothesis of all u in us
def inc_L0(us):
    res = 1
    for u in us:
        res *= MM[u]
    return res / np.sum(res)

# make the 1-step extension utterance
def inc_S0(w, us):
    past_utter_mask = np.ones((MM.shape[0],))
    past_utter_mask[us] = 0
    possible = MM[:, w] * past_utter_mask
    return possible

# make the incremental version of S1
def inc_SS1(w, us):
    u_news = np.where(inc_S0(w,us) == 1)[0]
    probs = np.zeros((MM.shape[0],))
    for u in u_news:
        new_utts = us + [u]
        probs[u] = inc_L0(new_utts)[w]
    return probs

# sample S1
def sample_S1(w, u_len):
    logpr = 0
    sampled_us = []
    for i in range(u_len):
        inc_u_prob = inc_SS1(w, sampled_us)
        u = np.argmax(inc_u_prob)
        sampled_us.append(u)
    return sampled_us    

# make the full S1 (this returns a logpr number)
def inc_S1(w, us):
    logpr = 0
    past_us = []
    for u in us:
        inc_u_prob = inc_SS1(w, past_us)
        inc_u_prob = inc_u_prob / np.sum(inc_u_prob)
        logpr += np.log(inc_u_prob[u])
        past_us.append(u)
    return logpr

def inc_L1(us):
    possible_ws = np.where(inc_L0(us) > 0)[0]
    ret = []
    for w in possible_ws:
        inc_pr = inc_S1(w, us)
        ret.append((inc_pr, w))
    ret.sort(key = lambda x: -x[0])
    return ret

# we can now measure communication effectiveness of pairs
# the process is as follows:
# for a given w in W (here a random reward function)
# select according to the speaker probability an utterance P_speak(u | w)
# give the utterance to the listener
# recover the w according to Plisten(w_recovered | u)
# measure probability P(w == w_recovered)
# this can be done analytically :
# compute P(w'= w) = integrate_u Pspeak(u | w) Plisten(w' | u)
def comm_acc(S,L):
    # how easy it is to communicate a particular hypothesis
    w_to_w = (S*L).sum(axis=0)
    # we simply take an average across all diff hypothesis
    return w_to_w.mean()

def test1():
    L0 = MM / MM.sum(axis=1)[:, np.newaxis]
    S0 = MM / MM.sum(axis=0)[np.newaxis, :]
    print (f"checking L0 row normalize to 1 {np.sum(L0[0])}")
    print (f"checking S0 col normalize to 1 {np.sum(S0[:, 0])}")
    S1 = L0 / L0.sum(axis=0)[np.newaxis, :]
    L1 = S1 / S1.sum(axis=1)[:, np.newaxis]
    print (f"checking S1 col normalize to 1 {np.sum(S1[:, 0])}")
    print (f"checking L1 row normalize to 1 {np.sum(L1[0])}")
    # now the real deal, is pragmatics helpful for this grid-world setting?
    print ("if we can only demonstrate ONE trajectory : ")
    print (f"communication effectiveness for S0-L0 pair {comm_acc(S0,L0)}")
    print (f"communication effectiveness for S1-L0 pair {comm_acc(S1,L0)}")
    print (f"communication effectiveness for S1-L1 pair {comm_acc(S1,L1)}")

if __name__ == '__main__':
    render_world(GRID, name='grid')
    print (W[100])
    u2s = sample_S1(100,2)
    print (u2s)
    for u in u2s:
        print (U[u])
    print (inc_L1(u2s))

    # competing_idx = [78,83,88,93,100]
    # for x in competing_idx:
    #     u2s = sample_S1(x,1)
    #     spk_pr = inc_S1(x, u2s)
    #     print (x, u2s, spk_pr)

    # print ("wait a sec")
    # print (inc_S1(100, [582]))
    # print (inc_S1(78, [582]))
    # print (inc_SS1(78, []))
    # print (inc_L1(u2s))
    # print (inc_SS1(100, [582]))
    # print (inc_S1(100, [582]))