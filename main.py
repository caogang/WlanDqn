from env.wlanenvironment import wlanEnv
from Brain import BrainDQN
import time

CONTROLLER_IP = '10.103.12.166:8080'
BUFFER_LEN = 60
ENV_REFRESH_INTERVAL = 0.1

def main():
    env = wlanEnv(CONTROLLER_IP, BUFFER_LEN, timeInterval=ENV_REFRESH_INTERVAL)
    env.start()

    numAPs, numActions = env.getDimSpace()
    brain = BrainDQN(numActions, numAPs, BUFFER_LEN)

    while not env.observe()[0]:
        time.sleep(0.5)

    observation0 = env.observe()[1]
    brain.setInitState(observation0)

    while True:
        action = brain.getAction()
        env.step(action)
        nextObservation = env.observe()
        reward = env.getReward()
        brain.setPerception(nextObservation, action, reward, False)
    


if __name__ == '__main__':
    main()