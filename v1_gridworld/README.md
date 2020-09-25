# What's In Here ?

run this to get a sense of the kind of problems we're solving

    python env.py
    
then run this to generate a meaning matrix for a _particular_ grid, fortunately the total space of utterances (different trajectories)  and hypothesis (different reward functions) are quite small, so it can be enumerated

    python build_meaning_matrix.py
    
this would dump a .npy file that is the cached meaning matrix

finally, run this to validate our hypothesis that pragmatics is indeed helpful. one niceness of pragmatics is that communication efficiency can be easily computed if you can represent the explicit matrix for S and L, and perform matrix operations on them

    python prag.py
    
