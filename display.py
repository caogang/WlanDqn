# -*- coding: utf-8 -*-
import matplotlib.pylab as plt
import numpy as np
import copy
import time
import threading
import pickle

class Display:
    def __init__(self, id2ap, REFRESH_INTERVAL=1, PREDICT=False):
        self.t_begin = None
        self.predict = PREDICT
        self.t = []
        self.rssi = {}
        self.action = []
        self.reward = []
        self.q_value = {}
        if self.predict:
            self.feature_vector = []

        self.end = False
        self.threads = []
        self.interval = REFRESH_INTERVAL
        self.id2ap = id2ap
        pass

    def append(self, data):
        """
        Pass a dict to display which is in a specific format
        :param data:
        {'timestamp' : int, 'rssi' : np.array(numAps), 'q' : np.array(numAps), 'action_index' : int, 'reward' : int, 'feature_vector' : np.array(hidden_size) }
        :return:
        """
        t = copy.copy(self.t)
        q_value = copy.deepcopy(self.q_value)
        rssi = copy.deepcopy(self.rssi)
        reward = copy.copy(self.reward)
        action = copy.copy(self.action)
        if self.predict:
            feature_vector = copy.copy(self.feature_vector)

        if len(t) == 0:
            self.t_begin = data['timestamp']
            t.append(0)
        else:
            t.append(data['timestamp'] - self.t_begin)

        reward.append(data['reward'])
        action.append(data['action_index'])
        if self.predict:
            feature_vector.append(data['feature_vector'])

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
        if self.predict:
            self.feature_vector = copy.copy(feature_vector)

    def _plot(self):

        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True)
        rssi_fig = ax[0][0]
        reward_fig = ax[0][1]
        action_fig = ax[1][0]
        q_fig = ax[1][1]
        fig_tp = plt.figure()
	ax_feature_vector = fig_tp.add_subplot(111)

        while not self.end:

            t = copy.copy(self.t)
            q_value = copy.deepcopy(self.q_value)
            rssi = copy.deepcopy(self.rssi)
            reward = copy.copy(self.reward)
            action = copy.copy(self.action)
            if self.predict:
                feature_vector = copy.copy(self.feature_vector)
                feature_vector = np.array(feature_vector)
                feature_vector = feature_vector.transpose((1,0))
                print feature_vector.shape

            if len(t) == 0:
                time.sleep(self.interval)
                continue

            if len(t) != len(q_value[0]) or len(t) != len(rssi[0]) or len(t) != len(reward) or len(t) != len(action):
                continue

            max_t = t[-1] + 100
            min_t = max_t - 900 if max_t - 900 > 0 else 0

            rssi_fig.cla()
            rssi_fig.set_title("Rssi / dBm")
            rssi_fig.set_ylim(-90, -20)
            rssi_fig.set_xlim(min_t, max_t)
            rssi_fig.grid()
            for id, r in rssi.items():
                rssi_fig.plot(t, np.array(r) - 255, label=self.id2ap[id])
            rssi_fig.legend(loc='best')

            reward_fig.cla()
            reward_fig.set_title("Throughtout / Mbps")
            reward_fig.set_ylim(0, 70)
            reward_fig.set_xlim(min_t, max_t)
            reward_fig.grid()
            reward_fig.plot(t, reward)

            action_fig.cla()
            action_fig.set_title("AP")
            action_fig.set_xlabel("Time / s")
            action_fig.set_yticks([0, 1])
            action_fig.set_yticklabels([self.id2ap[i] for i in [0, 1]], rotation=30)
            action_fig.set_ylim(-1, 2)
            action_fig.set_xlim(min_t, max_t)
            action_fig.grid()
            action_fig.plot(t, action)

            q_fig.cla()
            q_fig.set_title("Q Value / Mbps")
            q_fig.set_xlabel("Time / s")
            q_fig.set_ylim(100, 160)
            q_fig.set_xlim(min_t, max_t)
            q_fig.grid()
            for id, q in q_value.items():
                q_fig.plot(t, q, label=self.id2ap[id])
            q_fig.legend(loc='best')

            #ax_feature_vector.imshow(feature_vector)

            plt.pause(self.interval)

        if self.predict:
            save_data = {
                't': self.t,
                'rssi': self.rssi,
                'q': self.q_value,
                'reward': self.reward,
                'action': self.action,
                'id2ap': self.id2ap,
                'feature_vector': self.feature_vector
            }
        else:
            save_data = {
                't': self.t,
                'rssi': self.rssi,
                'q': self.q_value,
                'reward': self.reward,
                'action': self.action,
                'id2ap': self.id2ap,
            }

        output = open('fig.pkl', 'wb')
        pickle.dump(save_data, output, -1)
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
