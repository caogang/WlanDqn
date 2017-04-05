import matplotlib.pylab as plt
import numpy as np
import copy
import time
import threading

class Display:
    def __init__(self, REFRESH_INTERVAL=1):
        self.t_begin = None
        self.t = []
        self.rssi = {}
        self.action = []
        self.reward = []
        self.q_value = {}

        self.end = False
        self.threads = []
        self.interval = REFRESH_INTERVAL
        pass

    def append(self, data):
        """
        Pass a dict to display which is in a specific format
        :param data:
        {'timestamp' : int, 'rssi' : np.array(numAps), 'q' : np.array(numAps), 'action_index' : int, 'reward' : int}
        :return:
        """
        t = copy.copy(self.t)
        q_value = copy.deepcopy(self.q_value)
        rssi = copy.deepcopy(self.rssi)
        reward = copy.copy(self.reward)
        action = copy.copy(self.action)

        if len(t) == 0:
            self.t_begin = data['timestamp']
            t.append(0)
        else:
            t.append(data['timestamp'] - self.t_begin)

        reward.append(data['reward'])
        action.append(data['action_index'])

        for ap, r in enumerate(data['rssi']):
            if ap not in rssi.keys():
                rssi[ap] = []
            rssi[ap].append(r)

        for ap, q in enumerate(data['q']):
            if ap not in q_value.keys():
                q_value[ap] = []
            q_value[ap].append(q)

        self.t = copy.copy(t)
        self.q_value = copy.deepcopy(q_value)
        self.rssi = copy.deepcopy(rssi)
        self.reward = copy.copy(reward)
        self.action = copy.copy(action)

    def _plot(self):

        fig, ax = plt.subplots(nrows=2, ncols=2)
        rssi_fig = ax[0][0]
        reward_fig = ax[0][1]
        action_fig = ax[1][0]
        q_fig = ax[1][1]

        while not self.end:
            if len(self.t) == 0:
                time.sleep(self.interval)
                continue
            rssi_fig.cla()
            rssi_fig.grid()
            rssi_fig.plot((self.t), self.rssi[0])

            reward_fig.cla()
            reward_fig.grid()
            reward_fig.plot((self.t), self.reward)

            action_fig.cla()
            action_fig.grid()
            action_fig.plot((self.t), self.action)

            q_fig.cla()
            q_fig.grid()
            q_fig.plot((self.t), self.q_value[0])

            plt.pause(self.interval)
        pass

    def display(self):
        t1 = threading.Thread(target=self._plot)
        self.threads.append(t1)
        for t in self.threads:
            t.setDaemon(True)
            t.start()
        print('display starting...')


    def stop(self):
        self.end = True
        for t in self.threads:
            t.join()
        print('stop display')


def display(data):
    """
    Pass a dict to display which is in a specific format
    :param data:
    {'timestamp' : int, 'rssi' : np.array(numAps), 'q' : np.array(numAps), 'action' : string, 'reward' : int}
    :return:
    """

    print(data)
    pass

if __name__ == '__main__':
    f = Display()
    f.display()

    data = {}
    data['timestamp'] = time.time()
    data['rssi'] = np.array([170, 108])
    data['q'] = np.array([142, 130])
    data['reward'] = 65
    data['action_index'] = 1

    f.append(data)
    time.sleep(0.1)

    data['timestamp'] = time.time()
    data['rssi'] = np.array([170, 108])
    data['q'] = np.array([142, 130])
    data['reward'] = 65
    data['action_index'] = 1

    f.append(data)
    time.sleep(0.1)

    data['timestamp'] = time.time()
    data['rssi'] = np.array([170, 108])
    data['q'] = np.array([142, 130])
    data['reward'] = 65
    data['action_index'] = 1

    f.append(data)
    time.sleep(0.1)

    data['timestamp'] = time.time()
    data['rssi'] = np.array([170, 108])
    data['q'] = np.array([142, 130])
    data['reward'] = 65
    data['action_index'] = 1

    f.append(data)
    time.sleep(0.1)
    # f.plot()

    time.sleep(5)
    f.stop()