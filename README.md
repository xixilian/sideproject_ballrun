# sideproject_ballrun

It's a side project of my master thesis ball run, using reinforcement with naive deep Q network to see how programms play marble run.

For the beginning, the simulation is reduced to 2D and using the similar idea for playing Atari.

The binary files "dynamics" and "track" may just work under linux environment or need to recompile for Mac OS. And it needs the ODE package compiled on the PC, to get ODE on your PC, please visit the ODE wiki page :
https://www.ode-wiki.org/wiki/index.php?title=Manual:_Install_and_Use

After get ODE compiled (!), some python dependencies need to be isntalled as well:

    tensorflow, numpy, scipy, matplotlib, Pillow, multimethod, glob
    
Most of them can be installed through : pip install ...

After all the dependencies stuffs, to train the network, simply type : python dqn.py
to the terminal.

And in the dqn.py file, there are parameters you can play with.

Have fun :D




