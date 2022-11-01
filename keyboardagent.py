import numpy as np
import cv2
import random
from env_carworld import *

env = CarWorld()
sensordivide = 61
sensorclass = 3
gamma = 0.98
horizon = 5000

def Reward(step):
    if agent.pos[0] >= 512. or agent.pos[0] <= -512. or agent.pos[1] >= 512. or agent.pos[1] <= -512. :
        return True, -1., "Out", 0.
    else:
        reward = 0.04 - ((21. - agent.vel) ** 2.) * 0.0025 * 0.02 - (agent.wheeltemp[0] + agent.wheeltemp[1] + agent.wheeltemp[2] + agent.wheeltemp[3]) * 0.02
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
    agent.Initialize()
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



agent = CarWorldCarWheel(env, sensorclass=sensorclass, sensorrangedeg = 120, sensordivide=sensordivide)
agent.vel = 0.
env.objects['agent'] = agent

balls = []
ball = CarWorldBall(0, (255, 0, 0))
balls.append(ball)
env.objects['ball' + str(len(balls))] = ball
'''
ball = CarWorldBall(1, (0, 255, 0))
balls.append(ball)
env.objects['ball' + str(len(balls))] = ball

for i in range(8):
    ball = CarWorldBall(2, (0, 0, 255))
    balls.append(ball)
    env.objects['ball' + str(len(balls))] = ball
'''

img = { 'center' : np.array([512, 512]), 'sizex' : 1024, 'sizey' : 1024 }

action = [1., 1., 1., 1. ]
while True:
    Env_Initialize()
    for step in range(horizon):
        env.Step()

        done, reward, msg, success = Reward(step)
        if done:
            break
        print("Reward : ", reward)

        state = np.concatenate([ [(horizon - step - 1) / horizon, agent.pos[0] / 512., agent.pos[1] / 512., np.cos(agent.orientation), np.sin(agent.orientation)], agent.inventory, agent.GetSensorData() ]).reshape(-1)
                

                  
        sensorimg = np.zeros((200, sensordivide * 20, 3), np.uint8)
        for i in range(sensordivide):
            color = (int(state[i*3+8] * 255), int(state[i*3+9] * 255), int(state[i*3+10] * 255))
            cv2.rectangle(sensorimg, (i*20, 0), (i*20+20, 200), color, -1)
        cv2.imshow('sensorimg', sensorimg)

        closeview = np.zeros((256, 256, 3), np.uint8)
        agent.DrawCloseView(closeview)
        cv2.imshow('closeview', closeview)
        
        img['image'] = np.zeros((1024, 1024, 3), np.uint8)
        env.Draw(img)
        cv2.imshow('image', img['image'])
        key = cv2.waitKey(20)

        if key & 0xFF == ord('q'):
            action = [-1., -1., 1., 1.]
        elif key & 0xFF == ord('w'):
            action = [0.5, 0.5, 0.5, 0.5]
        elif key & 0xFF == ord('e'):
            action = [1., -1., 1., 1.]
        elif key & 0xFF == ord('s'):
            action = [1., 1., 1., 1.]

        agent.ApplyAction(action)
        print(agent.vel, agent.orientation)