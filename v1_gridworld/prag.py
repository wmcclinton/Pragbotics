import numpy as np

MM = np.load("MeaningMatrix.npy")
L0 = MM / MM.sum(axis=1)[:, np.newaxis]
S0 = MM / MM.sum(axis=0)[np.newaxis, :]
print (f"checking L0 row normalize to 1 {np.sum(L0[0])}")
print (f"checking S0 col normalize to 1 {np.sum(S0[:, 0])}")
S1 = L0 / L0.sum(axis=0)[np.newaxis, :]
L1 = S1 / S1.sum(axis=1)[:, np.newaxis]
print (f"checking S1 col normalize to 1 {np.sum(S1[:, 0])}")
print (f"checking L1 row normalize to 1 {np.sum(L1[0])}")

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

# now the real deal, is pragmatics helpful for this grid-world setting?
print ("if we can only demonstrate ONE trajectory : ")
print (f"communication effectiveness for S0-L0 pair {comm_acc(S0,L0)}")
print (f"communication effectiveness for S1-L1 pair {comm_acc(S1,L1)}")


