from env.wlanenvironment import wlanEnv
from Brain import BrainDQN
import time
import numpy as np
from display import Display
# import signal

CONTROLLER_IP = '10.103.43.130:8080'
BUFFER_LEN = 60
ENV_REFRESH_INTERVAL = 0.1

# def sigint_handler(signum, frame):
#     """
#     Catch CTRL+C Event
#     :param signum:
#     :param frame:
#     :return:
#     """
#     global is_sigint_up
#     is_sigint_up = True
#     print 'Catched interrupt signal .'
#
# signal.signal(signal.SIGINT, sigint_handler)
# is_sigint_up = False

def train():
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

    data = {}
    fig = Display(env.id2ap)
    fig.display()

    try:
        while True:
            action, q = brain.getAction()
            print 'action:\n' + str(action.argmax())
            reward, throught, nextObservation = env.step(action)
            print 'reward: ' + str(reward) + ', throught: ' + str(throught)
            print 'Next observation:\n' + str(nextObservation)

            data['timestamp'] = time.time()
            data['rssi'] = nextObservation[-1]
            data['q'] = q
            data['reward'] = reward
            data['action_index'] = np.argmax(action)
            fig.append(data)

            brain.setPerception(nextObservation, action, reward, False)
    except KeyboardInterrupt:
        print 'Saving replayMemory......'
        brain.saveReplayMemory()
        fig.stop()
    pass

def test():
    env = wlanEnv(CONTROLLER_IP, BUFFER_LEN, timeInterval=ENV_REFRESH_INTERVAL, no_guarantee=True)
    env.start()

    numAPs, numActions, numAdditionDim = env.getDimSpace()
    brain = BrainDQN(numActions, numAPs, numAdditionDim, BUFFER_LEN, param_file='saved_networks/network-dqn.params')

    while not env.observe()[0]:
        time.sleep(0.5)

    observation = env.observe()[1]

    np.set_printoptions(threshold=5)

    data = {}
    fig = Display(env.id2ap)
    fig.display()
    try:
        while True:
            action, q_value, action_index = brain.predict(observation)
            print 'action:\n' + str(action_index)
            reward, throught, observation = env.step(action)
            print 'q_value: ' + str(q_value)
            print 'reward: ' + str(reward) + ', throught: ' + str(throught)
            data['timestamp'] = time.time()
            data['rssi'] = observation[-1]
            data['q'] = q_value
            data['reward'] = reward
            data['action_index'] = action_index
            fig.append(data)
            print 'Next observation:\n' + str(observation)
            time.sleep(2)
    except KeyboardInterrupt:
        fig.stop()

if __name__ == '__main__':
    test()
