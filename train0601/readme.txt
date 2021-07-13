SAC1 learn fellow

    SAC2 time 500

    DQN1=0531DQN2  speed[0.3, 0.5]
    DQN3_tune

    DQN2     "init_speed": tune.grid_search([[0.3,0.4],[0.4,0.5],[0.5,0.6],[0.6,0.7],[0.3,0.5],[0.4,0.6],[0.5,0.8]])
    DQN3     "init_speed": tune.grid_search([[0.3,0.4],[0.4,0.8],[0.3,0.7],[0.7,0.8]])
    DQN4     "     "init_speed": tune.grid_search([[0.3,0.4],[0.4,0.8],[0.3,0.7],[0.7,0.8]]),
    "exploration_fraction": tune.grid_search([0.4,0.8,0.95])"


todo 1.delate action
do state space tun

todo change the env!!!!!!!!!!!!!!!!!!!!
1) test the policy works in target env


todo 2 add the enaluation episiode reward