import threading
import urllib2
from time import sleep
import numpy as np
import json

class wlanEnv:
    def __init__(self, remoteControllerAddr, seqLen, timeInterval=0.1):
        self.remoteAddr = remoteControllerAddr
        self.numAp = 0
        self.seqLen = seqLen
        self.timeInverval = timeInterval
        self.threads = []
        self.end = False


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

    def __getStatesFromRemote(self, clientHwAddr, timeInterval):
        while not self.end:
            rssiUrl = 'http://' + self.remoteAddr + '/dqn/rssi/json?mac=' + clientHwAddr
            rssiDict = curl_keystone(rssiUrl)
            rssiDict = json.loads(rssiDict)
            rewardUrl = 'http://' + self.remoteAddr + '/dqn/reward/json?mac=' + clientHwAddr
            rewardDict = curl_keystone(rewardUrl)
            rewardDict = json.loads(rewardDict)
            # print 'rssi'
            # print rssiDict
            # print 'reward'
            # print rewardDict
            if rssiDict['state'] and rewardDict['state']:
                rssiDict.pop('state')
                rewardDict.pop('state')
                self.reward = rewardDict['reward']
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
            sleep(timeInterval)

    def __handover(self, clientHwAddr, agentIp):
        handoverUrl = 'http://' + self.remoteAddr + '/dqn/handover/json?mac=' + clientHwAddr + '&&agent=' + agentIp
        print handoverUrl

    '''
    @:returns
    input vector dimension
    action space dimension
    '''
    def getDimSpace(self):
        return self.numAp, self.numAp + 1

    def observe(self):
        rssi = self.obsevation.astype(int)
        return self.valid, rssi

    def step(self, action):
        actionId = action.index(1)
        if actionId >= self.numAp:
            return
        self.__handover(self.macAddr, self.id2ap[actionId])
        return

    def getReward(self):
        return self.valid, self.reward

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
    env.start()
    sleep(2)
    print env.observe()
    print env.step(0)
    sleep(0.1)
    print env.observe()
    print env.step(0)
    sleep(0.1)
    print env.observe()
    print env.step(0)
    sleep(0.1)
    print env.step(1)
    print env.getDimSpace()
    env.stop()
    sleep(2)
    pass