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
log_file = open(LOG_DIR + "log_test.txt", "wt")

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=0)
log_file.write("Episode\tScore\tStep\tQvalue\tQvalue_Std\tQvalue_Loss\n")

with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess, "data/dqn4/task_twogoal/log_5000.ckpt")
    mean_std = 1.
        
    for episode in range(1, 5001):
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

            for step in range(horizon):
                prev_state = state
                
                action, qvalue = learner.get_action_with_qvalue(state)
                print(qvalue)
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
                

                if done:
                    cur_message = msg
                    break


                img['image'] = np.zeros((1024, 1024, 3), np.uint8)
                env.Draw(img)
                cv2.imshow('image', img['image'])
                key = cv2.waitKey(0)
                    
            print("Batch " + str(cur_batch) + " Step " + str(step) + " Reward " + str(batch_reward) + " Inventory : " + str(agent.inventory) + " " + cur_message)

            if cur_message == "Success":
                cur_step += step
                cur_success_batch += 1

