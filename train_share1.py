import numpy as np
import cv2
import tensorflow as tf
import random
from env_carworld import *
from networks.dqn_share_learner import DQNSharedLearner
from networks.share_learner_selecter import DQNSelecter

sensordivide = 61
sensorclass = 3
gamma = 0.98
horizon = 250

class CarWorldEnv:
    def __init__(self, task, agent):
        self.env = CarWorld()
        self.agent = agent
        self.task = task
        self.balls = []
        self.env.objects['agent'] = agent
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
            reward = -0.02
        #    return True, -1., "Out"
        for val in self.balls:
            if val.draw:
                distance = np.sqrt( (val.pos[0] - self.agent.pos[0]) ** 2 + (val.pos[1] - self.agent.pos[1]) ** 2 )
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
        self.agent.pos = np.array([0, 0])
        self.agent.orientation = 0.
        self.agent.inventory = np.zeros(sensorclass)

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



agent = CarWorldCar(None, sensorclass=sensorclass, sensorrangedeg = 120, sensordivide=sensordivide)
agent.vel = 10.


img = { 'center' : np.array([512, 512]), 'sizex' : 1024, 'sizey' : 1024 }
learners = [DQNSharedLearner((sensordivide + 1) * sensorclass + 5, 3, name=str(i), gamma=gamma, hidden_len=64, embed_len=[32, 32])for i in range(12)]
selecter = DQNSelecter((sensordivide + 1) * sensorclass + 5 + 3, embed_len=1024)

LOG_DIR = "data/dqn_share1/train1/"
log_step_file = open(LOG_DIR + "log_step.txt", "wt")
log_result_file = open(LOG_DIR + "log_result.txt", "wt")

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=0)
log_step_file.write("Epoch\tTask\tBatch\tLearner\tReward\tStep\n")
log_result_file.write("Epoch\tTask\tReward\tMax_Learner\tFinal_Learner\n")


# (num of obejct x3) (reward x3) (steering addition x2) (automove x2)
tasks = [
    [ [ 1, 0, 8 ], [ 10., 0., -1.]],
    [ [ 1, 8, 0 ], [ 10., -1., 0.]],
    [ [ 8, 1, 0 ], [ -1., 10., 0.]],
    [ [ 0, 1, 8 ], [ 0., 10., -1.]],
    [ [ 0, 8, 1 ], [ 0., -1., 10.]],
    [ [ 8, 0, 1 ], [ -1., 0., 10.]]
]
    
envs = [CarWorldEnv(task, agent) for task in tasks]

with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)
    #saver.restore(sess, LOG_DIR + "log1.ckpt")
    mean_std = 1.
    mean_reward = [5.] * len(learners)

    cur_learner = 0

    for epoch in range(1, 10001):
        taski = random.randrange(6)
        cur_env = envs[taski]
        agent.env = cur_env.env

        state_vector = []
        next_state_vector = []
        action_vector = []
        reward_vector = []
        survive_vector = []
        value_vector = []
        pairs_vector = []
        vector_len = 0
        
        learner_index = [[]] * len(learners)
        final_reward = [0.] * len(learners)
        play_count = [0] * len(learners)
        reward_pooling = [15.] * len(learners)

        cur_embedding = [learner.get_embedding() for learner in learners]
        cur_reward = 0

        print("EPISODE " + str(epoch))
        for play in range(64):
            pair_vector = []
            cur_env.Env_Initialize()
            batch_reward = 0
            state = np.concatenate([ [1., agent.pos[0] / 512., agent.pos[1] / 512., np.cos(agent.orientation), np.sin(agent.orientation)], agent.inventory,  agent.GetSensorData() ]).reshape(-1)

            step_start = vector_len
            for step in range(horizon):
                prev_state = state
                
                action = learners[cur_learner].get_action_random(state, mean_std * 0.1)
                action_vec = np.zeros(3)
                action_vec[action] = 1.
                
                if action == 0:
                    agent.orientation -= 0.2
                elif action == 2:
                    agent.orientation += 0.2
                
                cur_env.env.Step()
                '''
                img['image'] = np.zeros((1024, 1024, 3), np.uint8)
                cur_env.env.Draw(img)
                cv2.imshow('image', img['image'])
                key = cv2.waitKey(1)
                '''
                done, reward = cur_env.Reward(step)

                state = np.concatenate([ [(horizon - step - 1) / horizon, agent.pos[0] / 512., agent.pos[1] / 512., np.cos(agent.orientation), np.sin(agent.orientation)], agent.inventory, agent.GetSensorData() ]).reshape(-1)

                state_vector.append(prev_state)
                pair_vector.append(np.concatenate([prev_state, action_vec]))
                action_vector.append(action_vec)
                survive_vector.append([ 0.0 if done else 1.0])
                next_state_vector.append(state)
                reward_vector.append([reward])
                value_vector.append([0.])
                
                learner_index[cur_learner].append(vector_len)

                vector_len += 1
                batch_reward += reward


                if done:
                    break

            discounted = 0.
            for i in range(vector_len - 1, step_start - 1, -1):
                if reward_vector[i][0] > 0.:
                    discounted = reward_vector[i][0]
                else:
                    discounted = discounted * gamma + reward_vector[i][0]
                value_vector[i][0] = discounted

            print("Task " + str(taski) + " Batch " + str(play) + "Learner " + str(cur_learner) + " Step " + str(step) + " Reward " + str(batch_reward))
            print("Mean reward : " + str(mean_reward[cur_learner]) + " Reward Pooling : " + str(reward_pooling[cur_learner]))
            log_step_file.write(str(epoch) + "\t" + str(taski) + "\t" + str(play) + "\t" + str(cur_learner) + "\t" + str(batch_reward) + "\t" + str(step) + "\n")

            cur_reward += batch_reward
            final_reward[cur_learner] += batch_reward
            reward_pooling[cur_learner] = reward_pooling[cur_learner] * 0.9 + batch_reward * 0.1

            play_count[cur_learner] += 1

            pred_embed = selecter.get_output(pair_vector)
            curmin = 1e15
            for i in range(len(learners)):
                if reward_pooling[i] > mean_reward[i]:
                    cur = np.mean((cur_embedding[i] - pred_embed) ** 2)
                    if cur < curmin:
                        curmin = cur
                        cur_learner = i
            pairs_vector.append(pair_vector)
        selecter.reset()

        curmax = -1e15
        maxlearner = 0
        for i in range(len(learners)):
            if play_count[i] != 0:
                final_reward[i] /= play_count[i]
                mean_reward[i] = mean_reward[i] * 0.95 + final_reward[i] * 0.05
                if mean_reward[i] < 5.:
                    mean_reward[i] = 5.
                if final_reward[i] > curmax:
                    curmax = final_reward[i]
                    maxlearner = i
        
        print("Task " + str(taski) + " Reward : " + str(np.max(final_reward)) + " Maximum Learner : " + str(maxlearner) + " Final Learner : " + str(cur_learner))
        log_result_file.write(str(epoch) + "\t" + str(taski) + "\t" + str(np.max(final_reward)) + "\t" + str(maxlearner) + "\t" + str(cur_learner) + "\n")

        state_vector_dic = [state_vector[x] for x in learner_index[maxlearner]]
        next_state_vector_dic = [next_state_vector[x] for x in learner_index[maxlearner]]
        action_vector_dic = [action_vector[x] for x in learner_index[maxlearner]]
        reward_vector_dic = [reward_vector[x] for x in learner_index[maxlearner]]
        survive_vector_dic = [survive_vector[x] for x in learner_index[maxlearner]]
        value_vector_dic = [value_vector[x] for x in learner_index[maxlearner]]

        q, qs, ql = learners[maxlearner].optimize_batch(state_vector_dic, next_state_vector_dic, action_vector_dic, reward_vector_dic, survive_vector_dic, value_vector_dic)
        mean_std = mean_std * 0.95 + qs * 0.05
        
        ls = 0
        lr = 0.00001
        for pair in pairs_vector:
            l = selecter.optimize(pair, learners[maxlearner].get_embedding(), lr)
            ls += l
            lr *= 1.1
        selecter.reset()
        print("Selector Loss : " + str(ls))

        if epoch % 1000 == 0:
            saver.save(sess, LOG_DIR + "log_" + str(epoch) + ".ckpt")
