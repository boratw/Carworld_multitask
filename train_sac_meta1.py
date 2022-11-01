import numpy as np
import cv2
import tensorflow as tf
import random
from env_carworld import *
from networks.sac_learner import SACLearner

sensordivide = 61
sensorclass = 3
gamma = 0.98
horizon = 200
action_dim = 4


class CarWorldEnv:
    def __init__(self, task):
        self.env = CarWorld()
        self.agent = CarWorldCarWheel(self.env, sensorclass=sensorclass, sensorrangedeg = 120, sensordivide=sensordivide, accratio=task[2])
        self.task = task
        self.balls = []
        self.env.objects['agent'] = self.agent
        for i in range(task[0][0]):
            ball = CarWorldBall(0, (255, 0, 0))
            self.balls.append(ball)
            self.env.objects['ball' + str(len(self.balls))] = ball
        for i in range(task[0][1]):
            ball = CarWorldBall(1, (0, 255, 0))
            self.balls.append(ball)
            self.env.objects['ball' + str(len(self.balls))] = ball
        for i in range(task[0][2]):
            ball = CarWorldBall(2, (0, 0, 255))
            self.balls.append(ball)
            self.env.objects['ball' + str(len(self.balls))] = ball

    def Reward(self, step):
        if self.agent.pos[0] >= 512. or self.agent.pos[0] <= -512. or self.agent.pos[1] >= 512. or self.agent.pos[1] <= -512. :
            return True, -1.
        else:
            reward = 0.04 - ((21. - (self.agent.vel if self.agent.vel < 20. else 1.)) ** 2.) * 0.0025 * 0.02 - (self.agent.wheeltemp[0] + self.agent.wheeltemp[1] + self.agent.wheeltemp[2] + self.agent.wheeltemp[3]) * 0.01
        for val in self.balls:
            if val.draw:
                distance = np.sqrt( (val.pos[0] - agent.pos[0]) ** 2 + (val.pos[1] - agent.pos[1]) ** 2 )
                if val.id == 0:
                    if distance < 24.:
                        return True, self.task[1][0]
                elif val.id == 1:
                    if distance < 24.:
                        return True, self.task[1][1]
                elif val.id == 2:
                    if distance < 24:
                        return True, self.task[1][2]
        if step == (horizon-1) :
            return True, 0.
        else:
            return False, reward

    def Env_Initialize(self):
        self.agent.Initialize()

        for i in range(len(self.balls)):
            self.balls[i].draw = True
            brk = True
            while brk :
                brk = False
                newpos = (np.random.rand(2) * 768. - 384.).astype(np.int32)
                if (newpos[0] ** 2 + newpos[1] ** 2) < 1024.:
                    brk = True
                for j in range(i):
                    if ((newpos[0] - self.balls[j].pos[0]) ** 2 + (newpos[1] - self.balls[j].pos[1]) ** 2) < 1024.:
                        brk = True
            self.balls[i].pos = newpos



img = { 'center' : np.array([512, 512]), 'sizex' : 1024, 'sizey' : 1024 }

tasks = [
    [ [ 1, 0, 8 ], [ 10., 0., -1.], [1., 1., 1., 1.]],
    [ [ 0, 1, 8 ], [ 0., 10., -1.], [1., 1., 1., 1.]],
    [ [ 1, 8, 0 ], [ 10., -1., 0.], [1., 1., 1., 1.]],
    [ [ 8, 1, 0 ], [ -1., 10., 0.], [1., 1., 1., 1.]],

    [ [ 1, 0, 8 ], [ 10., 0., -1.], [1., 1., 0., 1.]],
    [ [ 0, 1, 8 ], [ 0., 10., -1.], [1., 1., 0., 1.]],
    [ [ 1, 8, 0 ], [ 10., -1., 0.], [1., 1., 0., 1.]],
    [ [ 8, 1, 0 ], [ -1., 10., 0.], [1., 1., 0., 1.]],

    [ [ 1, 0, 8 ], [ 10., 0., -1.], [0., 1., 1., 1.]],
    [ [ 0, 1, 8 ], [ 0., 10., -1.], [0., 1., 1., 1.]],
    [ [ 1, 8, 0 ], [ 10., -1., 0.], [0., 1., 1., 1.]],
    [ [ 8, 1, 0 ], [ -1., 10., 0.], [0., 1., 1., 1.]]
]
    

learners = [SACLearner(sensordivide * sensorclass + 9, action_dim, gamma=gamma, name=str(i)) for i in range(len(tasks))]
envs = [CarWorldEnv(task) for task in tasks]

LOG_DIR = "data/sac_meta1/train1/"
sess = tf.Session()
saver = tf.train.Saver(max_to_keep=0)


with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)
    #saver.restore(sess, "data/dqn4/task1_zero/log_500.ckpt")

    for taskit in range(len(tasks)):
        log_file = open(LOG_DIR + "log_" + str(taskit) + ".txt", "wt")

        state_vector = []
        next_state_vector = []
        action_vector = []
        reward_vector = []
        value_vector = []
        survive_vector = []
        vector_len = 0

        learner = learners[taskit]
        env = envs[taskit]
        agent = env.agent

        learners[taskit].value_network_initialize()

        for episode in range(1, 10001):
            cur_reward = 0
            for cur_batch in range(8):
                env.Env_Initialize()
                force_break = 0
                force_break_count = 0
                batch_reward = 0
                reward_store = []
                step_start = vector_len

                state = np.concatenate([ [1., agent.pos[0] / 512., agent.pos[1] / 512., np.cos(agent.orientation), np.sin(agent.orientation)], agent.wheelvel,  agent.GetSensorData() ]).reshape(-1)
                for step in range(horizon):
                    prev_state = state
                    action = learner.get_action_walk(state)[0]
                    
                    if cur_batch > 3:
                        r = np.random.random()
                        if r < 0.005:
                            force_break = np.random.randint(4)
                            force_break_count = int(r * 1600 + 8)
                        if force_break_count > 0:
                            if force_break == 0:
                                action[0] = 0.5 + action[0] * 0.5
                                action[1] = 0.5 + action[1] * 0.5
                            elif force_break == 1:
                                action[0] = 0.5 + action[0] * 0.5
                                action[2] = 0.5 + action[2] * 0.5
                            elif force_break == 2:
                                action[1] = 0.5 + action[1] * 0.5
                                action[3] = 0.5 + action[3] * 0.5
                            else:
                                action[2] = 0.5 + action[2] * 0.5
                                action[3] = 0.5 + action[3] * 0.5
                            force_break_count -= 1
                    
                    agent.ApplyAction(action)
                    env.env.Step()

                    state = np.concatenate([ [(horizon - step - 1) / horizon, agent.pos[0] / 512., agent.pos[1] / 512., np.cos(agent.orientation), np.sin(agent.orientation)], agent.wheelvel,  agent.GetSensorData() ]).reshape(-1)
                    done, reward = env.Reward(step)

                    state_vector.append(prev_state)
                    action_vector.append(action)
                    survive_vector.append([ 0.0 if done else 1.0])
                    next_state_vector.append(state)
                    reward_vector.append([reward])
                    value_vector.append([0.0])

                    vector_len += 1
                    batch_reward += reward
                    if(done):
                        break

                    '''
                    closeview = np.zeros((256, 256, 3), np.uint8)
                    agent.DrawCloseView(closeview)
                    cv2.imshow('closeview', closeview)

                    if cur_batch == 0:
                        img['image'] = np.zeros((1024, 1024, 3), np.uint8)
                        env.Draw(img)
                        cv2.imshow('image', img['image'])
                        key = cv2.waitKey(1)
                    '''
                discounted = 0.
                for i in range(vector_len - 1, step_start - 1, -1):
                    if reward_vector[i][0] > discounted:
                        discounted = reward_vector[i][0]
                    else:
                        discounted = discounted * gamma + reward_vector[i][0]
                    value_vector[i][0] = discounted

                print("Batch " + str(cur_batch) + " Step " + str(step) + " Reward " + str(batch_reward))
                cur_reward += batch_reward

            ql = 0.
            vl = 0.
            pl = 0.
            for history in range(16):
                dic = random.sample(range(vector_len), 256)

                state_vector_dic = [state_vector[x] for x in dic]
                next_state_vector_dic = [next_state_vector[x] for x in dic]
                action_vector_dic = [action_vector[x] for x in dic]
                reward_vector_dic = [reward_vector[x] for x in dic]
                survive_vector_dic = [survive_vector[x] for x in dic]
                value_vector_dic = [value_vector[x] for x in dic]

                q, v, p = learner.optimize_batch(state_vector_dic, next_state_vector_dic, action_vector_dic, reward_vector_dic, survive_vector_dic, value_vector_dic)

                ql += np.mean(q)
                vl += np.mean(v)
                pl += np.mean(p)

            learner.value_network_update()

            ql /= 16.
            vl /= 16.
            pl /= 16.

            print("Epoch " + str(episode) + " Qvalue : " + str(ql) + " Value : " + str(vl) + " Policy : " + str(pl))


            log_file.write(str(episode) + "\t" +
                str(cur_reward / 8) + "\t"  + str(ql) + "\t" + str(vl) + "\t" + str(pl) + "\n")
            if episode % 1000 == 0:
                saver.save(sess, LOG_DIR + "log_" + str(taskit) + "_" + str(episode) + ".ckpt")

            vec_trunc = vector_len // 50
            state_vector = state_vector[vec_trunc:]
            next_state_vector = next_state_vector[vec_trunc:]
            action_vector = action_vector[vec_trunc:]
            reward_vector = reward_vector[vec_trunc:]
            survive_vector = survive_vector[vec_trunc:]
            value_vector = value_vector[vec_trunc:]


            vector_len -= vec_trunc