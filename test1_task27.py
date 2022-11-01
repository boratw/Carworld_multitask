import numpy as np
import cv2
import tensorflow as tf
import random
from env_carworld import *
from networks.dqn_ns_learner import DQNLearner

sensordivide = 61
sensorclass = 3
gamma = 0.98
horizon = 500

def Reward(step):
    if agent.pos[0] >= 512. or agent.pos[0] <= -512. or agent.pos[1] >= 512. or agent.pos[1] <= -512. :
        return True, -1.
    else:
        reward = -0.01
    #    return True, -1., "Out"
    for val in env.objects.values():
        if val.name == 'ball' and val.draw:
            distance = np.sqrt( (val.pos[0] - agent.pos[0]) ** 2 + (val.pos[1] - agent.pos[1]) ** 2 )
            if val.id == 0:
                if distance < 24.:
                    return True, cur_task[1][0]
            elif val.id == 1:
                if distance < 24.:
                    return True, cur_task[1][1]
            elif val.id == 2:
                if distance < 24:
                    return True, cur_task[1][2]
    if step == (horizon-1) :
        return True, 0.
    else:
        return False, reward

def Env_Initialize():
    agent.pos = np.array([0, 0])
    agent.orientation = 0.
    agent.inventory = np.zeros(sensorclass)

    for i in range(len(balls)):
        balls[i].draw = True
        brk = True
        while brk :
            brk = False
            newpos = (np.random.rand(2) * 768. - 384.).astype(np.int32)
            if (newpos[0] ** 2 + newpos[1] ** 2) < 1024.:
                brk = True
            for j in range(i):
                if ((newpos[0] - balls[j].pos[0]) ** 2 + (newpos[1] - balls[j].pos[1]) ** 2) < 1024.:
                    brk = True
        balls[i].pos = newpos



agent = CarWorldCar(None, sensorclass=sensorclass, sensorrangedeg = 120, sensordivide=sensordivide)
agent.vel = 10.


img = { 'center' : np.array([512, 512]), 'sizex' : 1024, 'sizey' : 1024 }
learners = [DQNLearner((sensordivide + 1) * sensorclass + 5, 3, name=str(i), qvalue_lr=0.001, gamma=gamma) for i in range(27)]

LOG_DIR = "data/dqn_task27/basic_train/"
log_file = open(LOG_DIR + "log_test.txt", "wt")

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=0)
log_file.write("Task\tLearner\tReward\tScore_s\tScore_v\n")


# (num of obejct x3) (reward x3) (steering addition x2) (automove x2)
tasks = [
    [ [ 0, 1, 8 ], [ 0., 10., -1.], [0.2, -0.2], [0.0, 0.0] ],
    [ [ 0, 8, 1 ], [ 0., -1., 10.], [0.2, -0.2], [0.0, 0.0] ],
    [ [ 1, 8, 0 ], [ 10., -1., -1.], [0.2, -0.2], [0.0, 0.0] ],
    [ [ 0, 1, 8 ], [ 0., 10., -1.], [0.05, -0.2], [0.0, 0.0] ],
    [ [ 0, 8, 1 ], [ 0., -1., 10.], [0.05, -0.2], [0.0, 0.0] ],
    [ [ 1, 8, 0 ], [ 10., -1., -1.], [0.05, -0.2], [0.0, 0.0] ],
    [ [ 0, 1, 8 ], [ 0., 10., -1.], [0.2, -0.05], [0.0, 0.0] ],
    [ [ 0, 8, 1 ], [ 0., -1., 10.], [0.2, -0.05], [0.0, 0.0] ],
    [ [ 1, 8, 0 ], [ 10., -1., -1.], [0.2, -0.05], [0.0, 0.0] ],

    [ [ 0, 1, 8 ], [ 0., 10., -1.], [0.2, -0.2], [0.0, 2.5] ],
    [ [ 0, 8, 1 ], [ 0., -1., 10.], [0.2, -0.2], [0.0, 2.5] ],
    [ [ 1, 8, 0 ], [ 10., -1., -1.], [0.2, -0.2], [0.0, 2.5] ],
    [ [ 0, 1, 8 ], [ 0., 10., -1.], [0.05, -0.2], [0.0, 2.5] ],
    [ [ 0, 8, 1 ], [ 0., -1., 10.], [0.05, -0.2], [0.0, 2.5] ],
    [ [ 1, 8, 0 ], [ 10., -1., -1.], [0.05, -0.2], [0.0, 2.5] ],
    [ [ 0, 1, 8 ], [ 0., 10., -1.], [0.2, -0.05], [0.0, 2.5] ],
    [ [ 0, 8, 1 ], [ 0., -1., 10.], [0.2, -0.05], [0.0, 2.5] ],
    [ [ 1, 8, 0 ], [ 10., -1., -1.], [0.2, -0.05], [0.0, 2.5] ],
    
    [ [ 0, 1, 8 ], [ 0., 10., -1.], [0.2, -0.2], [0.0, -2.5] ],
    [ [ 0, 8, 1 ], [ 0., -1., 10.], [0.2, -0.2], [0.0, -2.5] ],
    [ [ 1, 8, 0 ], [ 10., -1., -1.], [0.2, -0.2], [0.0, -2.5] ],
    [ [ 0, 1, 8 ], [ 0., 10., -1.], [0.05, -0.2], [0.0, -2.5] ],
    [ [ 0, 8, 1 ], [ 0., -1., 10.], [0.05, -0.2], [0.0, -2.5] ],
    [ [ 1, 8, 0 ], [ 10., -1., -1.], [0.05, -0.2], [0.0, -2.5] ],
    [ [ 0, 1, 8 ], [ 0., 10., -1.], [0.2, -0.05], [0.0, -2.5] ],
    [ [ 0, 8, 1 ], [ 0., -1., 10.], [0.2, -0.05], [0.0, -2.5] ],
    [ [ 1, 8, 0 ], [ 10., -1., -1.], [0.2, -0.05], [0.0, -2.5] ]
]
    


with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess, LOG_DIR + "log1.ckpt")

    
    for taski, cur_task in enumerate(tasks):

        env = CarWorld()
        env.objects['agent'] = agent
        agent.env = env
        balls = []

        for i in range(cur_task[0][0]):
            ball = CarWorldBall(0, (255, 0, 0))
            balls.append(ball)
            env.objects['ball' + str(len(balls))] = ball
        for i in range(cur_task[0][1]):
            ball = CarWorldBall(1, (0, 255, 0))
            balls.append(ball)
            env.objects['ball' + str(len(balls))] = ball
        for i in range(cur_task[0][2]):
            ball = CarWorldBall(2, (0, 0, 255))
            balls.append(ball)
            env.objects['ball' + str(len(balls))] = ball

        for learneri, learner in enumerate(learners):

            cur_reward = 0
            cur_score_v = 0
            cur_score_s = 0
            cur_step = 0

            for play in range(32):

                reward_vector = []
                value_vector = []

                Env_Initialize()
                batch_reward = 0
                batch_score_v = 0.
                batch_score_s = 0.
                state = np.concatenate([ [1., agent.pos[0] / 512., agent.pos[1] / 512., np.cos(agent.orientation), np.sin(agent.orientation)], agent.inventory,  agent.GetSensorData() ]).reshape(-1)

                for step in range(horizon):
                    prev_state = state
                    
                    action, qvalues = learner.get_action_with_qvalue(state)
                    action_vec = np.zeros(3)
                    action_vec[action] = 1.
                    
                    state_est = learner.get_next_state(state, action_vec)

                    if action == 0:
                        agent.orientation += cur_task[2][0]
                    elif action == 2:
                        agent.orientation += cur_task[2][1]
                    
                    env.Step()
                    agent.pos[0] += cur_task[3][0]
                    agent.pos[1] += cur_task[3][1]


                    done, reward = Reward(step)

                    state = np.concatenate([ [(horizon - step - 1) / horizon, agent.pos[0] / 512., agent.pos[1] / 512., np.cos(agent.orientation), np.sin(agent.orientation)], agent.inventory, agent.GetSensorData() ]).reshape(-1)
                    
                    batch_score_s += np.mean((state - state_est) ** 2)
                    batch_reward += reward
                    reward_vector.append([reward])
                    value_vector.append([qvalues[action]])
                    


                    if done:
                        break

                discounted = 0.
                for i in range(len(reward_vector) - 1, -1, -1):
                    if reward_vector[i][0] > 0.:
                        discounted = reward_vector[i][0]
                    else:
                        discounted = discounted * gamma + reward_vector[i][0]
                    batch_score_v += (value_vector[i][0] - discounted) ** 2

                cur_reward += batch_reward
                cur_score_v += batch_score_v
                cur_score_s += batch_score_s

            cur_reward /= 32
            cur_score_v /= 32
            cur_score_s /= 32

            print("Task " + str(taski) + "Learner " + str(learneri) + " Reward : " + str(cur_reward) + " Score_S : " + str(cur_score_s) + " Score_V : " + str(cur_score_v))

            log_file.write(str(taski) + "\t" + str(learneri) + "\t" +
                str(cur_reward) + "\t" + str(cur_score_s) +  "\t" + str(cur_score_v) + "\n")