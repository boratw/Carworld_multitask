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
log_file = open(LOG_DIR + "log2_4.txt", "wt")

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=0)
log_file.write("Learner\tEpisode\tReward\tScore_s\tScore_v\tQvalue\tQvalue_Std\tQvalue_Loss\tState_Loss\n")


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
    mean_std = 1.

    cur_task = tasks[16]

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

    for learneri in [22, 25]:

        learner = learners[learneri]
        state_vector = []
        next_state_vector = []
        action_vector = []
        reward_vector = []
        survive_vector = []
        value_vector = []
        vector_len = 0



        for episode in range(1, 301):

            cur_added = 0
            cur_success_batch = 0
            cur_reward = 0
            cur_score_v = 0
            cur_score_s = 0
            cur_step = 0
            for play in range(8):
                Env_Initialize()
                batch_reward = 0
                batch_score_v = 0.
                batch_score_s = 0.
                state = np.concatenate([ [1., agent.pos[0] / 512., agent.pos[1] / 512., np.cos(agent.orientation), np.sin(agent.orientation)], agent.inventory,  agent.GetSensorData() ]).reshape(-1)

                step_start = vector_len
                for step in range(horizon):
                    prev_state = state
                    
                    action, qvalues = learner.get_action_random_with_qvalue(state, mean_std * 0.1)
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

                    state_vector.append(prev_state)
                    action_vector.append(action_vec)
                    survive_vector.append([ 0.0 if done else 1.0])
                    next_state_vector.append(state)
                    reward_vector.append([reward])
                    value_vector.append([qvalues[action]])
                    
                    vector_len += 1
                    cur_added += 1
                    batch_reward += reward
                    batch_score_s += np.mean((state - state_est) ** 2)


                    if done:
                        break

                discounted = 0.
                for i in range(vector_len - 1, step_start - 1, -1):
                    if reward_vector[i][0] > 0.:
                        discounted = reward_vector[i][0]
                    else:
                        discounted = discounted * gamma + reward_vector[i][0]
                    batch_score_v += (value_vector[i][0] - discounted) ** 2
                    value_vector[i][0] = discounted


                print("Batch " + str(play) + " Step " + str(step) + " Reward " + str(batch_reward) + " Inventory : " + str(agent.inventory))

                if reward > 0.:
                    cur_step += step
                    cur_success_batch += 1

                cur_reward += batch_reward
                cur_score_v += batch_score_v
                cur_score_s += batch_score_s

            cur_reward /= 8
            cur_score_v /= 8
            cur_score_s /= 8

            q1 = 0.
            qs1 = 0.
            ql1 = 0.
            sl1 = 0.
            for history in range(32):
                dic = random.sample(range(vector_len), 256 if vector_len > 256 else vector_len)

                state_vector_dic = [state_vector[x] for x in dic]
                next_state_vector_dic = [next_state_vector[x] for x in dic]
                action_vector_dic = [action_vector[x] for x in dic]
                reward_vector_dic = [reward_vector[x] for x in dic]
                survive_vector_dic = [survive_vector[x] for x in dic]
                value_vector_dic = [value_vector[x] for x in dic]

                q, qs, ql, sl = learner.optimize_batch(state_vector_dic, next_state_vector_dic, action_vector_dic, reward_vector_dic, survive_vector_dic, value_vector_dic)

                q1 += q
                qs1 += qs
                ql1 += ql
                sl1 += sl

            learner.network_update()

            q1 /= 32.
            qs1 /= 32.
            ql1 /= 32.
            sl1 /= 32.
            mean_std = mean_std * 0.95 + qs1 * 0.05
            print("Epoch " + str(episode) + " Score_S : " + str(cur_score_s) + " Score_V : " + str(cur_score_v))


            log_file.write(str(learneri) + "\t" + str(episode) + "\t" +
                str(cur_reward) + "\t" + str(cur_score_s) + "\t" + str(cur_score_v)+  "\t" + str(q1) + "\t" + str(qs1) + "\t" + str(ql1) + "\t" + str(sl1) + "\n")

            vec_trunc = vector_len // 50
            state_vector = state_vector[vec_trunc:]
            next_state_vector = next_state_vector[vec_trunc:]
            action_vector = action_vector[vec_trunc:]
            reward_vector = reward_vector[vec_trunc:]
            survive_vector = survive_vector[vec_trunc:]
            value_vector = value_vector[vec_trunc:]

            vector_len -= vec_trunc

#saver.save(sess, LOG_DIR + "log1.ckpt")