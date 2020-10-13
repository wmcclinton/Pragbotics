from env import *
from prag import *
from build_meaning_matrix import *

def interactive(target):
    print ("you are communicating ")
    print (list(zip(C,W[target])))
    possible_uss = MM[:,target]
    possible_trs = dict()
    for uid in range(len(possible_uss)):
        if possible_uss[uid] > 0:
            possible_trs[U[uid]] = uid 
    us = []
    scenarios = []

    while True:
        print ("trajectories demonstrated so-far : ")
        for j, scen in enumerate(scenarios):
            print (f"senario {j}, {scen}")

        print ("choose a scenario of starting position ")
        inp = input("start x y is >>")
        x,y = [int(_) for _ in inp.split(' ')]
        print (f"starting from x={x},y={y}, there are these trajectories")
        trajs = [key for key in possible_trs.keys() if key[0] == (x,y)]
        for i, tr in enumerate(trajs):
            print (f"traj id {i}, trajectory {tr}")
        inp_trj = int(input("pick a specific trajectory to show. traj id is >>"))
        print (f"added trajectory {trajs[inp_trj]} to demonstration")
        scenarios.append(trajs[inp_trj])
        uid = possible_trs[trajs[inp_trj]]
        us.append(uid)

        result = inc_L1(us)
        print (f"top 3 guesses (out of {len(result)})")
        for logpr, r in result[:3]:
            print (f"{W[r]} logpr {logpr}")

if __name__ == '__main__':
    # import pdb; pdb.set_trace()
    interactive(48)
