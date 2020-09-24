import env
import numpy as np

GRID = np.array([[1,0,0],
       [1,0,2],
       [0,2,1],])

# grab the space of all hypothesis
W = env.enum_R()
# grab the space of all utterances
U = []
for key in env.ALL_TR:
    U += env.ALL_TR[key]

# build the meaning matrix if running this file 
if __name__ == '__main__':
    MeaningMatrix = np.zeros((len(U), len(W)))
    for r_idx, R in enumerate(W):
        print (f"getting opt trajectory for hypothesis {r_idx} of {len(W)}")
        for start_loc in [(x,y) for x in range(env.L) for y in range(env.L)]:
            best_trajs_from_start = env.get_best_trajs(GRID, R, start_loc)
            for tr in best_trajs_from_start:
                u_idx = U.index(tr)
                MeaningMatrix[u_idx][r_idx] = 1

    save_path = "MeaningMatrix.npy"
    print (f"meaning matrix shape {MeaningMatrix.shape}")
    print (f"total number of 1s in matrix {np.sum(MeaningMatrix)}")
    np.save(save_path, MeaningMatrix)
    print (f"saved {save_path} .")