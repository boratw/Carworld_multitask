import numpy as np
import cv2
import tensorflow as tf
import random
from env_carworld import *
from networks.dqn_learner import DQNLearner

env = CarWorld()
sensordivide = 61
sensorclass = 3
gamma = 0.98
horizon = 500

def Reward(step):
    if agent.pos[0] >= 512. or agent.pos[0] <= -512. or agent.pos[1] >= 512. or agent.pos[1] <= -512. :
        return True, -1., "Out", 0.
    else:
        reward = -0.01
    #    return True, -1., "Out"
    for val in env.objects.values():
        if val.name == 'ball' and val.draw:
            distance = np.sqrt( (val.pos[0] - agent.pos[0]) ** 2 + (val.pos[1] - agent.pos[1]) ** 2 )
            if val.id == 0:
                if distance < 24.:
                    if agent.inventory[1] == 1.:
                        return True, 10., "Success", 10.
            if val.id == 1:
                if distance < 24.:
                    agent.inventory[1] = 1.
                    val.draw = False
                    return False, 10., "GetKey", 10.
            elif val.id == 2:
                if distance < 24:
                    return True, -1., "Dead", 0.
                if distance < 96:
                    reward -= (1. / distance)
    if step == (horizon-1) :
        return True, 0., "TimeOut", 0.
    else:
        return False, reward, "", 0.

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



agent = CarWorldCar(env, sensorclass=sensorclass, sensorrangedeg = 120, sensordivide=sensordivide)
agent.vel = 10.
env.objects['agent'] = agent

balls = []
ball = CarWorldBall(0, (255, 0, 0))
balls.append(ball)
env.objects['ball' + str(len(balls))] = ball

ball = CarWorldBall(1, (0, 255, 0))
balls.append(ball)
env.objects['ball' + str(len(balls))] = ball

for i in range(8):
    ball = CarWorldBall(2, (0, 0, 255))
    balls.append(ball)
    env.objects['ball' + str(len(balls))] = ball


img = { 'center' : np.array([512, 512]), 'sizex' : 1024, 'sizey' : 1024 }
learner = DQNLearner((sensordivide + 1) * sensorclass + 5, 3, qvalue_lr=0.001, gamma=gamma)

LOG_DIR = "data/dqn4/task_twogoal/"
log_file = open(LOG_DIR + "log.txt", "wt")

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=0)
log_file.write("Episode\tScore\tStep\tQvalue\tQvalue_Std\tQvalue_Loss\n")

state_vector = []
next_state_vector = []
action_vector = []
reward_vector = []
survive_vector = []
value_vector = []
vector_len = 0

with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)
    #saver.restore(sess, "data/dqn4/task1_zero/log_500.ckpt")
    mean_std = 1.
        
    for episode in range(1, 5001):
        '''
        if episode % 200 == 0:
            ball = CarWorldBall(1, (0, 0, 255))
            balls.append(ball)
            env.objects['ball' + str(len(balls))] = ball
        '''
        cur_added = 0
        cur_batch = 0
        cur_success_batch = 0
        cur_reward = 0
        cur_step = 0
        while cur_added < 3000:
            Env_Initialize()
            batch_reward = 0
            state = np.concatenate([ [1., agent.pos[0] / 512., agent.pos[1] / 512., np.cos(agent.orientation), np.sin(agent.orientation)], agent.inventory,  agent.GetSensorData() ]).reshape(-1)
            cur_message = "TimeOut"

            step_start = len(value_vector)
            for step in range(horizon):
                prev_state = state
                
                action = learner.get_action_random(state, mean_std * 0.1)
                #action = random.randrange(3)
                
                if action == 0:
                    agent.orientation += 0.2
                elif action == 2:
                    agent.orientation -= 0.2
                

                env.Step()
                #agent.pos = np.clip(agent.pos, -512, 512)


                done, reward, msg, success = Reward(step)
                action_vec = np.zeros(3)
                action_vec[action] = 1.

                state = np.concatenate([ [(horizon - step - 1) / horizon, agent.pos[0] / 512., agent.pos[1] / 512., np.cos(agent.orientation), np.sin(agent.orientation)], agent.inventory, agent.GetSensorData() ]).reshape(-1)
                
                state_vector.append(prev_state)
                action_vector.append(action_vec)
                survive_vector.append([ 0.0 if done else 1.0])
                next_state_vector.append(state)
                reward_vector.append([reward])
                value_vector.append([0.0])
                
                vector_len += 1
                cur_added += 1
                batch_reward += reward

                if success != 0.:
                    for i in range(step_start, vector_len):
                        des_value = success * gamma ** (vector_len - i)
                        if value_vector[i][0] < des_value:
                             value_vector[i][0] = des_value


                if done:
                    cur_message = msg
                    break


                if cur_batch == 0:
                    img['image'] = np.zeros((1024, 1024, 3), np.uint8)
                    env.Draw(img)
                    cv2.imshow('image', img['image'])
                    key = cv2.waitKey(1)
                    '''
                    if key & 0xFF == ord('q'):
                        agent.orientation += 0.2
                    elif key & 0xFF == ord('w'):
                        agent.orientation -= 0.2

                    print(agent.inventory)                    
                    sensorimg = np.zeros((200, sensordivide * 20, 3), np.uint8)
                    for i in range(sensordivide):
                        color = (int(state[i*3+8] * 255), int(state[i*3+9] * 255), int(state[i*3+10] * 255))
                        cv2.rectangle(sensorimg, (i*20, 0), (i*20+20, 200), color, -1)
                    cv2.imshow('sensorimg', sensorimg)
                    '''
                    
            print("Batch " + str(cur_batch) + " Step " + str(step) + " Reward " + str(batch_reward) + " Inventory : " + str(agent.inventory) + " " + cur_message)

            if cur_message == "Success":
                cur_step += step
                cur_success_batch += 1
            '''
            if cur_message == "Success":
                step_iter = step
                while step_iter >= 10:
                    state_vector.extend(state_vector[-step_iter:])
                    next_state_vector.extend(next_state_vector[-step_iter:])
                    action_vector.extend(action_vector[-step_iter:])
                    reward_vector.extend(reward_vector[-step_iter:])
                    survive_vector.extend(survive_vector[-step_iter:])
                    vector_len += step_iter
                    step_iter = step_iter // 2
            '''

            cur_reward += batch_reward
            cur_batch += 1


        q1 = 0.
        qs1 = 0.
        ql1 = 0.
        for history in range(32):
            dic = random.sample(range(vector_len), 256)

            state_vector_dic = [state_vector[x] for x in dic]
            next_state_vector_dic = [next_state_vector[x] for x in dic]
            action_vector_dic = [action_vector[x] for x in dic]
            reward_vector_dic = [reward_vector[x] for x in dic]
            survive_vector_dic = [survive_vector[x] for x in dic]
            value_vector_dic = [value_vector[x] for x in dic]

            q, qs, ql = learner.optimize_batch(state_vector_dic, next_state_vector_dic, action_vector_dic, reward_vector_dic, survive_vector_dic, value_vector_dic)

            q1 += q
            qs1 += qs
            ql1 += ql

        learner.network_update()

        q1 /= 32.
        qs1 /= 32.
        ql1 /= 32.
        mean_std = mean_std * 0.95 + qs1 * 0.05
        print("Epoch " + str(episode) + " Qvalue : " + str(q1) + " Std : " + str(qs1) + " Loss : " + str(ql1))


        log_file.write(str(episode) + "\t" +
            str(cur_reward / cur_batch) + "\t" + str(cur_step / cur_success_batch if cur_success_batch != 0 else 0.) +  "\t" + str(q1) + "\t" + str(qs1) + "\t" + str(ql1) + "\n")
        if episode % 200 == 0:
            saver.save(sess, LOG_DIR + "log_" + str(episode) + ".ckpt")

        vec_trunc = vector_len // 50
        state_vector = state_vector[vec_trunc:]
        next_state_vector = next_state_vector[vec_trunc:]
        action_vector = action_vector[vec_trunc:]
        reward_vector = reward_vector[vec_trunc:]
        survive_vector = survive_vector[vec_trunc:]
        value_vector = value_vector[vec_trunc:]

        vector_len -= vec_trunc