import threading
import urllib2
from time import sleep
import numpy as np
import json
import time
import math

class wlanEnv:
    def __init__(self, remoteControllerAddr, seqLen):
        self.remoteAddr = remoteControllerAddr
        self.numAp = 0
        self.seqLen = seqLen
        self.threads = []
        self.end = False
        self.timeRewardMax = 10  # FIXME: let it be a parameter
        self.startTime = None

        self.macAddr = '94:65:9c:84:a3:32'
        rssiUrl = 'http://' + self.remoteAddr + "/dqn/rssi/json?mac=" + self.macAddr
        rssiDict = curl_keystone(rssiUrl)
        rssiDict = json.loads(rssiDict)
        dictKey = rssiDict.keys()
        dictKey.remove('state')
        self.numAp = len(dictKey)
        # self.ap2id = dict(zip(dictKey, xrange(0, self.numAp)))
        self.id2ap = dict(zip(xrange(0, self.numAp), dictKey))
        self.obsevation = None
        self.valid = False

        state = self.observe()[0]
        while state is False:
            state = self.observe()[0]

    def __calculateTimeReward(self):
        if self.startTime is None:
            self.startTime = time.time()
        lastTime = time.time() - self.startTime
        p = 0
        if lastTime >= 10:
            p = 1
        else:
            lastTime = lastTime * 3 / 10
            lastTime = lastTime - 3
            # print lastTime
            p = (math.exp(lastTime) - math.exp(-lastTime)) / (math.exp(lastTime) + math.exp(-lastTime))
            p = p + 1
            # print p
        return p * self.timeRewardMax

    def cal(self):
        return self.__calculateTimeReward()

    def __handover(self, clientHwAddr, agentIp):
        handoverUrl = 'http://' + self.remoteAddr + '/dqn/handover/json?mac=' + clientHwAddr + '&&agent=' + agentIp
        print handoverUrl
        curl_keystone(handoverUrl)

    '''
    @:returns
    input vector dimension
    action space dimension
    '''
    def getDimSpace(self):
        return self.numAp, self.numAp + 1

    def observe(self):
        rssiUrl = 'http://' + self.remoteAddr + '/dqn/rssi/json?mac=' + self.macAddr
        rssiDict = curl_keystone(rssiUrl)
        rssiDict = json.loads(rssiDict)
        if rssiDict['state']:
            rssiDict.pop('state')
            if self.obsevation is None:
                self.obsevation = np.array([rssiDict.values()])
            elif self.obsevation.shape[0] == self.seqLen:
                obsevation = np.delete(self.obsevation, (0), axis=0)
                obsevation = np.append(obsevation, [rssiDict.values()], axis=0)
                self.obsevation = obsevation
                if not self.valid:
                    self.valid = True
            else:
                self.obsevation = np.append(self.obsevation, [rssiDict.values()], axis=0)

        rssi = self.obsevation.astype(int)
        return self.valid, rssi

    def step(self, action):
        actionId = action.argmax()
        if actionId < self.numAp:
            self.__handover(self.macAddr, self.id2ap[actionId])
            self.startTime = time.time()

        _, reward, throught = self.getReward()
        _, nextObservation = self.observe()

        return reward, throught, nextObservation

    def getReward(self):
        rewardUrl = 'http://' + self.remoteAddr + '/dqn/reward/json?mac=' + self.macAddr
        rewardDict = curl_keystone(rewardUrl)
        rewardDict = json.loads(rewardDict)
        rewardDict.pop('state')
        throught = rewardDict['reward']
        reward = rewardDict['reward'] + self.__calculateTimeReward()

        return self.valid, reward, throught


def curl_keystone(url):
    req = urllib2.Request(url)
    response = urllib2.urlopen(req)
    return response.read()

if __name__ == '__main__':
    env = wlanEnv('10.103.12.166:8080', 10)
    print env.observe()
    print env.step(np.array([0,0,1]))
    # print env.cal()
    # sleep(1)
    # print env.cal()
    # sleep(1)
    # print env.cal()
    # sleep(1)
    # print env.cal()
    # sleep(1)
    # print env.cal()
    # sleep(1)
    # print env.cal()
    # sleep(1)
    # print env.cal()
    # sleep(1)
    # print env.cal()
    '''
    env.start()
    sleep(2)
    print env.observe()
    print env.step([1,0,0])
    sleep(0.1)
    print env.observe()
    print env.step([1,0,0])
    sleep(0.1)
    print env.observe()
    print env.step([1,0,0])
    sleep(0.1)
    print env.step([0,1,0])
    print env.getDimSpace()
    env.stop()
    sleep(2)
    '''
    pass
