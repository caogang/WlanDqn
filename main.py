from env.wlanenvironment import wlanEnv
from Brain import BrainDQN
import time

CONTROLLER_IP = '10.103.12.166:8080'
BUFFER_LEN = 60

def main():
    env = wlanEnv(CONTROLLER_IP, BUFFER_LEN)

    numAPs, numActions = env.getDimSpace()
    brain = BrainDQN(numActions, numAPs, BUFFER_LEN, param_file='saved_networks/network-dqn-8900.params')

    observation0 = env.observe()[1]
    brain.setInitState(observation0)
    while True:
        action = brain.getAction()
        print 'action:\n' + str(action.argmax())
        reward, throught, nextObservation = env.step(action)
        print 'reward:\n' + str(reward)
        print 'throught: ' + str(throught)
        brain.setPerception(nextObservation, action, reward, False)
    


if __name__ == '__main__':
    main()
