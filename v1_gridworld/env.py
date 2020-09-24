import random
import numpy as np

# a simple LxL grid world
L = 3
# each grid is one of 3 colours as below
# in code we just use color code 0, 1, 2 instead of strings
C = ["red", "blue", "green"] #, "yellow"]
def make_random_world():
    return np.random.randint(3, size=(L, L))

# the possible rewards for each color, range from -1 to 1
C_R = (-2,3)

# make the reward function
def make_R():
    ret = [random.choice(range(*C_R)) for x in C]
    random.shuffle(ret)
    return ret

# enumerate all reward function
def enum_R():
    def enum_R_rec(n):
        if n == 0:
            return [()]
        else:
            rest = enum_R_rec(n-1)
            ret = []
            for i in range(*C_R):
                for r in rest:
                    ret.append((i,) + r)
            return ret
    return enum_R_rec(len(C))


# given a world dimension, enumerate ALL possible trajectories
# that starts at the start_location
# rules : 1) you cannot visit a same cell twice
#         2) you cannot go off the grid
def get_all_trajectory(start_loc):

    def get_next_locs(cur_loc, used_loc):
        x,y = cur_loc
        proposals = [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
        legal = lambda xy: 0<=xy[0]<L and 0<=xy[1]<L and xy not in used_loc
        return list(filter(legal, proposals))

    def all_tra_recursive(prefix):
        cur_loc = prefix[-1]
        nxt_locs = get_next_locs(cur_loc, prefix)
        if nxt_locs == []:
            return [prefix]
        else:
            ret = []
            for nxt_loc in nxt_locs:
                extended_prefix = prefix + (nxt_loc,)
                rest = all_tra_recursive(extended_prefix)
                ret += rest
            ret += [prefix]
            return ret

    return all_tra_recursive((start_loc,))

ALL_TR = dict()
for start_loc in [(x,y) for x in range(L) for y in range(L)]:
    ALL_TR[start_loc] = get_all_trajectory(start_loc)

# given a trajectory, get the score of that trejectory
def get_score(grid, R, traj,debug=False):
    ret = 0
    for (x,y) in traj:
        color_id = grid[x][y]
        # a base score of -1 at every step to penalize long trajectories
        ret += R[color_id] - 1
        if debug:
            print (C[color_id])
            print (ret)
    return ret

def get_best_trajs(grid, R, start_loc):
    all_tr = ALL_TR[start_loc]
    all_scores =  [get_score(grid, R, tr) for tr in all_tr]
    best_score = max(all_scores)
    best_idxs = [idx for idx in range(len(all_scores)) if all_scores[idx] == best_score]
    return [all_tr[best_id] for best_id in best_idxs]

def render_world(grid, name='world'):
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    plt.figure()
    currentAxis = plt.gca()
    currentAxis.set_aspect('equal')
    for i in range(L):
        for j in range(L):
            color_idx = world[i][j]
            color = C[color_idx]
            currentAxis.add_patch(Rectangle((i/L, j/L), 1/L, 1/L,facecolor=color,edgecolor='black'))
    
    plt.savefig(f'drawings/{name}.png')


if __name__ == '__main__':
    start_pos = (2,0)
    all_traj = get_all_trajectory(start_pos)
    world = make_random_world()
    R = make_R()
    all_scores = [get_score(world, R, traj) for traj in all_traj]
    top_score_id = np.argmax(all_scores)
    top_traj = all_traj[top_score_id]
    print (world)
    render_world(world)
    print (list(zip(C,R)))
    print (top_traj, all_scores[top_score_id])
    print (get_score(world, R, top_traj, debug=True))

    best_tr_0_0 = get_best_trajs(world, R, (0,0))
    print (best_tr_3_3)
    print (len(best_tr_3_3))
