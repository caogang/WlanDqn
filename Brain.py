# -----------------------------
# File: Deep Q-Learning Algorithm
# Author: Flood Sung
# Date: 2016.3.21
# -----------------------------

import mxnet as mx
import numpy as np
import random
from collections import deque
from env.wlanenvironment import wlanEnv
import time

# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 100.  # timesteps to observe before training
EXPLORE = 200000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.001  # 0.001 # final value of epsilon
INITIAL_EPSILON = 0.1  # 0.01 # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
UPDATE_TIME = 100

RNN_NUMS_LAYER = 1
RNN_NUMS_HIDDEN = 512
ctx = mx.cpu()


class BrainDQN:
    def __init__(self, numActions, numAps, seqLen, param_file=None):
        # init replay memory
        self.replayMemory = deque()
        # init some parameters
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.numActions = numActions
        self.numAps = numAps
        self.seqLen = seqLen

        self.target = self.createQNetwork(isTrain=False)

        self.Qnet = self.createQNetwork()
        if param_file != None:
            self.Qnet.load_params(param_file)
        self.copyTargetQNetwork()
        # saving and loading networks

    def sym(self, predict=False):
        stack = mx.rnn.SequentialRNNCell()
        for i in range(RNN_NUMS_LAYER):
            stack.add(mx.rnn.LSTMCell(num_hidden=RNN_NUMS_HIDDEN, prefix='lstm_l%d_' % i))

        data = mx.sym.Variable('data')
        yInput = mx.sym.Variable('yInput')
        actionInput = mx.sym.Variable('actionInput')

        stack.reset()
        outputs, states = stack.unroll(self.seqLen, inputs=data, merge_outputs=True)
        
        if predict :
            pred = mx.sym.Reshape(outputs, shape=(1, -1))
        else:
            pred = mx.sym.Reshape(outputs, shape=(BATCH_SIZE, -1))

        fc1 = mx.sym.FullyConnected(data=pred, num_hidden=512, name='fc1')
        relu4 = mx.sym.Activation(data=fc1, act_type='relu', name='relu4')
        Qvalue = mx.sym.FullyConnected(data=relu4, num_hidden=self.numActions, name='qvalue')
        temp = Qvalue * actionInput
        coeff = mx.sym.sum(temp, axis=1, name='temp1')
        output = (coeff - yInput) ** 2
        loss = mx.sym.MakeLoss(output)

        if predict:
            return Qvalue
        else:
            return loss

    def createQNetwork(self, bef_args=None, isTrain=True):
        if isTrain:
            modQ = mx.mod.Module(symbol=self.sym(), data_names=('data', 'actionInput'), label_names=('yInput',),
                                 context=ctx)
            batch = BATCH_SIZE
            modQ.bind(data_shapes=[('data', (batch, self.seqLen, self.numAps)), ('actionInput', (batch, self.numActions))],
                      label_shapes=[('yInput', (batch,))],
                      for_training=isTrain)

            modQ.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34), arg_params=bef_args)
            modQ.init_optimizer(
                optimizer='adam',
                optimizer_params={
                    'learning_rate': 0.0002,
                    'wd': 0.,
                    'beta1': 0.5,
                })
        else:
            modQ = mx.mod.Module(symbol=self.sym(predict=True), data_names=('data',), label_names=None, context=ctx)
            batch = 1
            modQ.bind(data_shapes=[('data', (batch, self.seqLen, self.numAps))],
                      for_training=isTrain)

            modQ.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34), arg_params=bef_args)

        return modQ

    def copyTargetQNetwork(self):
        arg_params, aux_params = self.Qnet.get_params()
        # arg={}
        # for k,v in arg_params.iteritems():
        #    arg[k]=arg_params[k].asnumpy()
        # print arg_params, aux_params
        self.target.init_params(initializer=None, arg_params=arg_params, aux_params=aux_params, force_init=True)

        # args,auxs=self.target.get_params()
        # arg1={}
        # for k,v in args.iteritems():
        #    arg1[k]=args[k].asnumpy()
        print 'time to copy'

    def trainQNetwork(self):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = np.squeeze([data[0] for data in minibatch])
        action_batch = np.squeeze([data[1] for data in minibatch])
        reward_batch = np.squeeze([data[2] for data in minibatch])
        nextState_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = np.zeros((BATCH_SIZE,))
        Qvalue = []
        for i in range(BATCH_SIZE):
            self.target.forward(mx.io.DataBatch([mx.nd.array(nextState_batch[i].reshape(1, self.seqLen, self.numAps), ctx)], []))
            Qvalue.append(self.target.get_outputs()[0].asnumpy())
        Qvalue_batch = np.squeeze(Qvalue)
        terminal = np.squeeze([data[4] for data in minibatch])
        y_batch[:] = reward_batch
        if (terminal == False).shape[0] > 0:
            y_batch[terminal == False] += (GAMMA * np.max(Qvalue_batch, axis=1))[terminal == False]

        self.Qnet.forward(mx.io.DataBatch([mx.nd.array(state_batch, ctx), mx.nd.array(action_batch, ctx)],
                                          [mx.nd.array(y_batch, ctx)]), is_train=True)
        self.Qnet.backward()
        self.Qnet.update()

        # save network every 1000 iteration
        if self.timeStep % 100 == 0:
            self.Qnet.save_params('saved_networks/network-dqn_mx%04d.params' % (self.timeStep))

        if self.timeStep % UPDATE_TIME == 0:
            self.copyTargetQNetwork()

    def setInitState(self, observation):
        self.currentState = observation

    def setPerception(self, nextObservation, action, reward, terminal):
        # newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)

        self.replayMemory.append((self.currentState, action, reward, nextObservation, terminal))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE:
            # Train the network
            self.trainQNetwork()

        # print info
        state = ""
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print "TIMESTEP", self.timeStep, "/ STATE", state, \
            "/ EPSILON", self.epsilon

        self.currentState = nextObservation
        self.timeStep += 1

    def getAction(self, retIndex=False):
        # print type(self.currentState)
        self.target.forward(mx.io.DataBatch([mx.nd.array(self.currentState.reshape(1, self.seqLen, self.numAps), ctx)], []))
        QValue = np.squeeze(self.target.get_outputs()[0].asnumpy())
        action = np.zeros(self.numActions)
        action_index = 0
        if self.timeStep > OBSERVE and self.timeStep % FRAME_PER_ACTION == 0:
            ran = random.random()
            print 'random: ' + str(ran)
            if ran <= self.epsilon:
                action_index = random.randrange(self.numActions)
                action[action_index] = 1
            else:
                action_index = np.argmax(QValue)
                action[action_index] = 1
        else:
            action[self.numActions-1] = 1  # do nothing

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        if retIndex:
            return action_index
        else:
            # print 'type return action :' + str(type(action))
            return action

if __name__ == '__main__':
    CONTROLLER_IP = '10.103.12.166:8080'
    BUFFER_LEN = 3
    ENV_REFRESH_INTERVAL = 0.1
    env = wlanEnv(CONTROLLER_IP, BUFFER_LEN, timeInterval=ENV_REFRESH_INTERVAL)
    env.start()

    numAPs, numActions = env.getDimSpace()
    brain = BrainDQN(numActions, numAPs, BUFFER_LEN)

    while not env.observe()[0]:
        time.sleep(0.5)

    observation0 = env.observe()[1]
    brain.setInitState(observation0)
    action = brain.getAction()
    env.step(action)
    nextObservation = env.observe()
    reward = env.getReward()
    brain.setPerception(nextObservation, action, reward, False)
    env.stop()
