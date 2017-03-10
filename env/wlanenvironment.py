import threading
import urllib2
from time import sleep

class wlanEnv:
    def __init__(self, remoteControllerAddr, seqLen, timeInterval):
        self.remoteAddr = remoteControllerAddr
        self.numAp = 0 #FIXME: make this work
        self.seqLen = seqLen
        self.timeInverval = timeInterval
        self.threads = []
        self.end = False;

    def getStatesFromRemote(self, clientHwAddr, timeInterval):
        while not self.end:
            rssiUrl = 'http://' + self.remoteAddr + "/dqn/rssi/json?mac=" + clientHwAddr
            rssiDict = curl_keystone(rssiUrl)
            rewardUrl = 'http://' + self.remoteAddr + "/dqn/reward/json?mac=" + clientHwAddr
            rewardDict = curl_keystone(rewardUrl)
            print 'rssi'
            print rssiDict
            print 'reward'
            print rewardDict
            sleep(timeInterval)

    def start(self):
        t1 = threading.Thread(target=self.getStatesFromRemote, args=('94:65:9c:84:a3:32', self.timeInverval))
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
    env = wlanEnv('10.103.12.166:8080', 10, timeInterval=1)
    env.start()
    sleep(5)
    env.stop()
    sleep(2)
    pass