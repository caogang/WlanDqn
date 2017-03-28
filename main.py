from env.wlanenvironment import wlanEnv
from Brain import BrainDQN
import time
import numpy as np

CONTROLLER_IP = '10.103.12.166:8080'
BUFFER_LEN = 60
ENV_REFRESH_INTERVAL = 0.1

def main():
    env = wlanEnv(CONTROLLER_IP, BUFFER_LEN, timeInterval=ENV_REFRESH_INTERVAL)
    env.start()

    numAPs, numActions, numAdditionDim = env.getDimSpace()
    brain = BrainDQN(numActions, numAPs, numAdditionDim, BUFFER_LEN, param_file='saved_networks/network-dqn.params')

    while not env.observe()[0]:
        time.sleep(0.5)

    observation0 = env.observe()[1]
    brain.setInitState(observation0)

    np.set_printoptions(threshold=5)
    print 'Initial observation:\n' + str(observation0)

    while True:
        action = brain.getAction()
        print 'action:\n' + str(action.argmax())
        reward, throught, nextObservation = env.step(action)
        print 'reward: ' + str(reward) + ', throught: ' + str(throught)
        print 'Next observation:\n' + str(nextObservation)
        brain.setPerception(nextObservation, action, reward, False)

if __name__ == '__main__':
    main()
