import numpy as np
import cv2


class CarWorld:
    def __init__(self):
        self.objects = {}

    def Draw(self, img):
        for val in self.objects.values():
            val.Draw(img)

    def Step(self):
        for val in self.objects.values():
            if val.movable:
                val.pos[0] -= val.vel * np.sin(val.orientation)
                val.pos[1] += val.vel * np.cos(val.orientation)


    
class CarWorldObject:
    def __init__(self):
        self.pos = np.array([0., 0.])
        self.vel = 0.
        self.orientation = 0.
        self.movable = False
        self.name = ''
    
    def Draw(self, img):
        raise NotImplementedError
    

class CarWorldCar(CarWorldObject):
    def __init__(self, env, sensorclass=1, sensorrangedeg=120, sensordivide=41, sensordiffuserange=10, **kwargs):
        super().__init__()
        self.movable = True
        self.env = env
        self.name = 'car'
        self.sensorclass = sensorclass
        self.sensorratio = ( 180 / (sensorrangedeg / (sensordivide - 1)) ) / 3.1415926535
        self.sensordivide = sensordivide
        self.sensordiffuserange = sensordiffuserange
        self.inventory = None


    def Draw(self, img):
        pos = img['center'] + self.pos
        pts = np.array([ Rotate([-8.0, 8.0], self.orientation) + pos, Rotate([8.0, 8.0], self.orientation) + pos, 
                Rotate([8.0, -8.0], self.orientation) + pos, Rotate([-8.0, -8.0], self.orientation) + pos ], np.int32)
        cv2.fillPoly(img['image'], [pts], (255,128,64))
        pts = np.array([ Rotate([-2.0, 12.0], self.orientation) + pos, Rotate([2.0, 12.0], self.orientation) + pos, 
                Rotate([2.0, 0.0], self.orientation) + pos, Rotate([-2.0, 0.0], self.orientation) + pos ], np.int32)
        cv2.fillPoly(img['image'], [pts], (255,192,128))

    def GetSensorData(self):
        res = np.zeros((self.sensordivide, self.sensorclass), np.float32)
        if self.orientation > 3.141592653:
            self.orientation -= 6.283185307
        elif self.orientation < -3.141592653:
            self.orientation += 6.283185307
        halfdivide = (self.sensordivide - 1) // 2
            
        for val in self.env.objects.values():
            if val.name == 'ball' and val.draw:
                distance = np.sqrt( (val.pos[0] - self.pos[0]) ** 2 + (val.pos[1] - self.pos[1]) ** 2 )
                if distance < 16.:
                    distance = 16.
                ori = np.arctan2( val.pos[1] - self.pos[1], val.pos[0] - self.pos[0] ) - 1.570796327 - self.orientation
                if ori > 3.141592653:
                    ori -= 6.283185307
                elif ori < -3.141592653:
                    ori += 6.283185307
                oriint = np.around(ori * self.sensorratio)
                orideg = ori * self.sensorratio - oriint
                ori = int(oriint) + halfdivide
                dist = (48. / (distance + 32.))
                drop = dist * ((distance + 32.) / 48.) / (self.sensordiffuserange ** 2)
                for d in range(-self.sensordiffuserange, self.sensordiffuserange + 1):
                    if 0 <= ori + d < self.sensordivide:
                        distd = dist - drop * (orideg + d) ** 2
                        if res[ori + d, val.id] < distd :
                            res[ori + d, val.id] = distd

        return res.reshape(-1)

        def Initialize(self):
            self.pos = np.array([0., 0.])
            self.orientation = 0.
            self.vel = 0.
            self.inventory = np.zeros(self.sensorclass)
            

class CarWorldCarSteer(CarWorldCar):
    def __init__(self, env, **kwargs):
        super(CarWorldCarSteer, self).__init__(env, **kwargs)

    def ApplyAction(self, action):
        if action[0] < -1.:
            action[0] = -1.
        elif action[0] > 1.:
            action[0] = 1.

        if action[1] < -1.:
            action[1] = -1.
        elif action[1] > 1.:
            action[1] = 1.

        self.vel = self.vel + action[0] * 2.
        if self.vel < 0.:
            self.vel = 0.
        elif self.vel > 20.:
            self.vel = 20.

        self.orientation += action[1] * 0.2

    def Initialize(self):
        self.pos = np.array([0., 0.])
        self.orientation = 0.
        self.vel = 0.
        
class CarWorldCarWheel(CarWorldCar):
    def __init__(self, env, **kwargs):
        super(CarWorldCarWheel, self).__init__(env, **kwargs)
        self.wheeltemp = [0., 0., 0., 0.]
        self.wheelvel = [0., 0., 0., 0.]
        self.angvel = 0.
        self.accratio = kwargs["accratio"] if "accratio" in kwargs else [1., 1., 1., 1.] 
        self.brakeratio = kwargs["brakeratio"] if "brakeratio" in kwargs else [1., 1., 1., 1.] 

    def DrawCloseView(self, img):
        pts = np.array([ np.array([-48.0, -96.0]) + 128, np.array([-80.0, -96.0]) + 128, 
                np.array([-80.0, -32.0]) + 128, np.array([-48.0, -32.0]) + 128 ], np.int32)
        cv2.fillPoly(img, [pts], (0, 255, int(self.wheeltemp[0] * 512)) if self.wheeltemp[0] < 0.5 else (0, 511 - int(self.wheeltemp[0] * 511), 255))
        pts = np.array([ np.array([80.0, -96.0]) + 128, np.array([48.0, -96.0]) + 128, 
                np.array([48.0, -32.0]) + 128, np.array([80.0, -32.0]) + 128 ], np.int32)
        cv2.fillPoly(img, [pts], (0, 255, int(self.wheeltemp[1] * 512)) if self.wheeltemp[1] < 0.5 else (0, 511 - int(self.wheeltemp[1] * 511), 255))
        pts = np.array([ np.array([-48.0, 32.0]) + 128, np.array([-80.0, 32.0]) + 128, 
                np.array([-80.0, 96.0]) + 128, np.array([-48.0, 96]) + 128 ], np.int32)
        cv2.fillPoly(img, [pts], (0, 255, int(self.wheeltemp[2] * 512)) if self.wheeltemp[2] < 0.5 else (0, 511 - int(self.wheeltemp[2] * 511), 255))
        pts = np.array([ np.array([80.0, 32.0]) + 128, np.array([48.0, 32.0]) + 128, 
                np.array([48.0, 96.0]) + 128, np.array([80.0, 96.0]) + 128 ], np.int32)
        cv2.fillPoly(img, [pts], (0, 255, int(self.wheeltemp[3] * 512)) if self.wheeltemp[3] < 0.5 else (0, 511 - int(self.wheeltemp[3] * 511), 255))

        pts = np.array([ np.array([64.0, -64.0]) + 128, np.array([-64.0, -64.0]) + 128, 
                np.array([-64.0, 64.0]) + 128, np.array([64.0, 64.0]) + 128 ], np.int32)
        cv2.fillPoly(img, [pts], (255,128,64))
        pts = np.array([ np.array([16.0, -96.0]) + 128, np.array([-16.0, -96.0]) + 128, 
                np.array([-16.0, 0.0]) + 128, np.array([16.0, 0.0]) + 128 ], np.int32)
        cv2.fillPoly(img, [pts], (255,192,128))

    def ApplyAction(self, action):
        resistance = 0.001 * self.vel ** 2
        action = np.clip(action, -1., 1.)
        desacc = [ (x * 0.75 * self.accratio[i]  if x > 0. else x * (0.5 + self.vel * 0.025) * self.brakeratio[i]) for i, x in enumerate(action) ]
        desvel = [self.wheelvel[i] + desacc[i] for i in range(4)]

        frontacc = ((desacc[0] + desacc[1]) * 0.5 + (desacc[2] + desacc[3]) * 0.866025404) / 2.732050808 - resistance
        torque = (desacc[0] * 0.866025404 - desacc[1] * 0.866025404 + desacc[2] * 0.5 - desacc[3] * 0.5) / 2.732050808 / 11.313708499

        self.vel = self.vel + frontacc
        if self.vel < 0.:
            self.vel = 0.

        self.angvel = (self.angvel + torque) * 0.9
        if self.angvel < -1.570796327:
            self.angvel = -1.570796327
        elif self.angvel > 1.570796327:
            self.angvel = 1.570796327

        self.wheelvel[0] = self.vel + (self.angvel * 8. * 1.414213562 * 0.866025404)
        self.wheelvel[1] = self.vel - (self.angvel * 8. * 1.414213562 * 0.866025404)
        self.wheelvel[2] = self.vel + (self.angvel * 8. * 1.414213562 * 0.5)
        self.wheelvel[3] = self.vel - (self.angvel * 8. * 1.414213562 * 0.5)

        for i in range(4):
            self.wheeltemp[i] = (desacc[i] ** 2) * 0.05 + (abs(desvel[i] - self.wheelvel[i]) ** 2) * 2.5

            if self.wheeltemp[i] > 1.:
                self.wheeltemp[i] = 1.
            elif self.wheeltemp[i] < 0.:
                self.wheeltemp[i] = 0.

        self.orientation += self.angvel

    def Initialize(self):
        self.pos = np.array([0., 0.])
        self.orientation = 0.
        self.vel = 0.
        self.inventory = np.zeros(self.sensorclass)
        self.wheeltemp = [0., 0., 0., 0.]
        self.wheelvel = [0., 0., 0., 0.]

        




class CarWorldBall(CarWorldObject):
    def __init__(self, id, color):
        super().__init__()
        self.name = 'ball'
        self.id = id
        self.color = color
        self.show = True
    
    def Draw(self, img):
        if self.draw :
            pos = (img['center'] + self.pos)
            cv2.circle(img['image'], tuple(pos), 16, self.color, -1)


def Rotate(vec, ori):
    newx = vec[0] * np.cos(ori) - vec[1] * np.sin(ori)
    newy = vec[0] * np.sin(ori) + vec[1] * np.cos(ori)
    return np.array([newx, newy])
