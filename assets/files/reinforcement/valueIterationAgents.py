# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


# Mani Shah
# mshah22
# G00974705
# CS580
# Project 3
class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            newValues = self.values.copy()
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    v = [self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)]
                    newValues[state] = max(v)
            self.values = newValues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        sat = self.mdp.getTransitionStatesAndProbs(state, action)  # list of Sprime And T pairs
        total = 0
        for pair in sat:
            total += pair[1] * (self.mdp.getReward(state, action,pair[0]) + self.discount * self.values[pair[0]])
        return total

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        noLegal = [(float("-inf"), None)]
        return max(noLegal + [(self.computeQValueFromValues(state, action), action) for action in
                              self.mdp.getPossibleActions(state)], key=lambda x: x[0])[1]
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        for iteration in range(self.iterations):
            state = states[iteration % len(states)]
            if not self.mdp.isTerminal(state):
                self.values[state] = max([self.getQValue(state, action)
                                          for action in self.mdp.getPossibleActions(state)])

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        def getMaxQ(s):
            return max([self.computeQValueFromValues(s, a) for a in self.mdp.getPossibleActions(s)])

        def getPredecessors():
            predecessors = {}
            for s in self.mdp.getStates():
                if not self.mdp.isTerminal(s):
                    for a in self.mdp.getPossibleActions(s):
                        for tStateProb in self.mdp.getTransitionStatesAndProbs(s, a):
                            if tStateProb[0] in predecessors:
                                predecessors[tStateProb[0]].add(s)
                            else:
                                predecessors[tStateProb[0]] = {s}
            return predecessors

        queue = util.PriorityQueue()
        predecessors = getPredecessors()

        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                diff = abs(self.values[s] - getMaxQ(s))
                queue.push(s,-diff)

        for iteration in range(self.iterations):
            if queue.isEmpty():
                break
            s = queue.pop()
            if not self.mdp.isTerminal(s):
                self.values[s] = getMaxQ(s)
            for p in predecessors[s]:
                if not self.mdp.isTerminal(s):
                    diff = abs(self.values[p] - getMaxQ(p))
                    if diff > self.theta:
                        queue.update(p,-diff)