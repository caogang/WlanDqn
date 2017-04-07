import threading
import urllib2
from time import sleep
import numpy as np
import json
import time
import math

class wlanEnv:
    def __init__(self, remoteControllerAddr, seqLen, timeInterval=0.1, additionalDim=0, no_guarantee=False):
        self.remoteAddr = remoteControllerAddr
        self.numAp = 0
        self.seqLen = seqLen
        self.timeInverval = timeInterval
        self.threads = []
        self.end = False
        self.timeRewardMax = 5  # FIXME: let it be a parameter 5 10 20
        self.startTime = None
        self.no_guarantee = no_guarantee

        self.macAddr = '68:3e:34:9b:34:05'
        rssiUrl = 'http://' + self.remoteAddr + "/dqn/rssi/json?mac=" + self.macAddr
        rssiDict = curl_keystone(rssiUrl)
        rssiDict = json.loads(rssiDict)
        dictKey = rssiDict.keys()
        dictKey.remove('state')
        self.numAp = len(dictKey)
        self.ap2id = dict(zip(dictKey, xrange(0, self.numAp)))
        self.id2ap = dict(zip(xrange(0, self.numAp), dictKey))
        self.obsevation = None
        self.reward = None
        self.valid = False

        # initial actionId, currentId
        self.lastActionId = self.numAp
        self.currentId = self.__getCurrentId()
        self.additionalDim = additionalDim # FIXME: 0 or 2

    def __getCurrentId(self):
        url = 'http://' + self.remoteAddr + '/odin/clients/connected/json'
        dict = curl_keystone(url)
        # print dict
        dict = json.loads(dict)
        agentIp = dict[self.macAddr]['agent']
        agentId = self.ap2id[agentIp]
        return agentId

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
        if self.additionalDim > 0:
            return p * self.timeRewardMax
        else:
            return 0  # FIXME : need to be tanh(X) X>0

    def cal(self):
        return self.__calculateTimeReward()

    def __getStatesFromRemote(self, clientHwAddr, timeInterval):
        while not self.end:
            try:
                rssiUrl = 'http://' + self.remoteAddr + '/dqn/rssi/json?mac=' + clientHwAddr
                rssiDict = curl_keystone(rssiUrl)
                rssiDict = json.loads(rssiDict)
                rewardUrl = 'http://' + self.remoteAddr + '/dqn/reward/json?mac=' + clientHwAddr
                rewardDict = curl_keystone(rewardUrl)
                rewardDict = json.loads(rewardDict)
            except:
                print 'Error or Exception in __getStatesFromRemote()'
            else:
                # print 'rssi'
                # print rssiDict
                # print 'reward'
                # print rewardDict
                if len(rssiDict) == (self.numAp + 1) and len(rewardDict) == 2:
                    if rssiDict['state'] and rewardDict['state']:
                        rssiDict.pop('state')
                        rewardDict.pop('state')

                        if self.obsevation is None :
                            self.obsevation = np.array([rssiDict.values()])
                        elif self.obsevation.shape[0] == self.seqLen:
                            obsevation = np.delete(self.obsevation, (0), axis=0)
                            obsevation = np.append(obsevation, [rssiDict.values()],axis=0)
                            self.obsevation = obsevation
                            if not self.valid:
                                self.valid = True
                        else:
                            self.obsevation = np.append(self.obsevation, [rssiDict.values()], axis=0)

                        if self.reward is None:
                            self.reward = np.array([rewardDict['reward']])
                        elif self.reward.shape[0] == (self.seqLen//2):
                            reward = np.delete(self.reward, (0), axis=0)
                            reward = np.append(reward, [rewardDict['reward']], axis=0)
                            self.reward = reward
                        else:
                            self.reward = np.append(self.reward, [rewardDict['reward']], axis=0)
                else:
                    print "Some ap is not working......Please check!!!"
            finally:
                sleep(timeInterval)

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
        if self.additionalDim > 0:
            return self.numAp, self.numAp+1, self.additionalDim
        else:
            return self.numAp, self.numAp, self.additionalDim

    def observe(self):
        rssi = self.obsevation.astype(int)
        if self.additionalDim > 0:
            addition = np.array([self.lastActionId, self.currentId])
            return self.valid, (rssi, addition)
        else:
            return self.valid, rssi

    def step(self, action):
        actionId = action.argmax()
        if (self.additionalDim > 0 and actionId < self.numAp) or (self.additionalDim <= 0 and actionId != self.currentId):
            self.__handover(self.macAddr, self.id2ap[actionId])
            if not self.no_guarantee:
                sleep(self.timeInverval * self.seqLen)
            self.currentId = actionId
            self.startTime = time.time()

        _, reward, throught = self.getReward()
        self.lastActionId = actionId
        _, nextObservation = self.observe()

        return reward, throught, nextObservation

    def getReward(self):
        self.throught = self.reward.mean()
        # print self.throught, self.reward.std()
        return self.valid, self.throught, self.throught

    def start(self):
        t1 = threading.Thread(target=self.__getStatesFromRemote, args=(self.macAddr, self.timeInverval))
        self.threads.append(t1)
        for t in self.threads:
            t.setDaemon(True)
            t.start()
        print 'start'

    def stop(self):
        self.end = True
        for t in self.threads:
            t.join()
        print 'stop'

def curl_keystone(url):
    req = urllib2.Request(url)
    response = urllib2.urlopen(req)
    return response.read()

if __name__ == '__main__':
    env = wlanEnv('10.103.12.166:8080', 10, timeInterval=0.1)
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
    env.start()
    sleep(2)
    print env.step(np.array([1,0,0]))
    sleep(1)
    print env.step(np.array([0,0,1]))
    sleep(1)
    print env.step(np.array([0,0,1]))
    sleep(1)
    print env.step(np.array([0,1,0]))
    sleep(1)
    print env.step(np.array([0,0,1]))
    sleep(1)
    print env.step(np.array([0,0,1]))
    sleep(1)
    print env.step(np.array([1,0,0]))
    sleep(1)
    print env.step(np.array([0,0,1]))
    sleep(1)
    print env.step(np.array([0,0,1]))
    sleep(1)
    env.stop()
    sleep(2)
    pass
