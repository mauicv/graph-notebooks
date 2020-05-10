from itertools import tee
import random
import numpy as np
import matplotlib.pyplot as plt
import math


# For generating pairwise elements in a list. [1,2,3,4,5] becomes (1,2),
# (2,3), (3,4), (4,5)
def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class Walker:
    """For running random walks on the graph."""

    def __init__(self, state):
        self.state = state


class MetropolisWalker:
    """For running metropolos hastings algorithm on the graph."""

    def __init__(self, state):
        self.state = state


class State:
    """Abstaction of single state such as in the Walker class to a
    distrubution of states."""

    def __init__(self, states):
        self.states = {state: 0 for state in states}

    def __str__(self):
        string = ''
        count = 0
        print('--------------------------------------------------------------')
        for i, k in self.states.items():
            count = count + 1
            string = string + '{}  :  {:.5f}  |  '.format(i, k)
            if count > 5:
                string = string + '\n'
                count = 0.
        return string

    def draw(self):
        objects = self.states.keys()
        y_pos = np.arange(len(objects))
        performance = [val for k, val in self.states.items()]
        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('density')
        plt.title('Word occurence orbit frequency')
        plt.show()

    @classmethod
    def from_orbit(cls, orbit):
        instance = cls(orbit)
        for point in orbit:
            instance.states[point] = instance.states.get(point, 0) + 1
        for key, value in instance.states.items():
            instance.states[key] = instance.states[key]/len(orbit)
        return instance

    @classmethod
    def from_uniform(cls, states):
        instance = cls(states)
        for key, value in instance.states.items():
            instance.states[key] = 1/len(instance.states)
        return instance

    def __sub__(self, other):
        difference = State(self.states.keys())
        for key, _ in difference.states.items():
            difference.states[key] = abs(self.states.get(key, 0)
                                         - other.states.get(key, 0))
        return difference

    def dist(self, other):
        diff = other - self
        return math.sqrt(sum([v**2 for _, v in diff.states.items()]))


class TransitionMatrix:
    """Going to map between states with this class. """
    def __init__(self, states, data=None):
        self.p = {state: {} for state in states}
        if data is None:
            data = states
        self._count(data)
        self._normalize()

    def _count(self, states):
        for letter_1, letter_2 in pairwise(states):
            self.p[letter_1][letter_2] = self.p[letter_1].get(letter_2, 0) + 1

    def _normalize(self):
        for key, val in self.p.items():
            row_total = sum([count for _, count in val.items()])
            for target, count in val.items():
                self.p[key][target] = self.p[key][target]/row_total

    def T(self):
        """computes transpose."""

        newt = TransitionMatrix([state for state, _
                                 in self.p.items()], data=[])
        for s_1, P in self.p.items():
            for s_2, p in P.items():
                newt.p[s_2][s_1] = p
        return newt

    def __matmul__(self, other):
        """If applying the transistion matrix class to a Walker class then
        select the next state at random. If applying to a State class we
        generate a new distrbution. If a MetropolisWalker then it updates
        the state with respect to the acceptance probability."""

        if isinstance(other, State):
            new_state = State([s for s, _ in other.states.items()])
            for s_1, p in other.states.items():
                sum_p_s2_s1 = 0
                for s_2, P in self.p.items():
                    sum_p_s2_s1 = sum_p_s2_s1 + other.states[s_2]*P.get(s_1, 0)
                new_state.states[s_1] = sum_p_s2_s1
            return new_state

        if isinstance(other, Walker):
            ps = self.p[other.state]
            choices = [*ps.keys()]
            weights = [*ps.values()]
            choice = random.choices(choices, weights=weights).pop()
            return Walker(choice)

        if isinstance(other, MetropolisWalker):
            ps = self.p[other.state]
            choices = [*ps.keys()]
            weights = [*ps.values()]
            choice = random.choices(choices, weights=weights).pop()

            p_c_s = self.p[choice][other.state]
            p_s_c = self.p[other.state][choice]
            p_a = random.random()

            if p_a < min(1, p_c_s/p_s_c):
                return MetropolisWalker(choice)
            else:
                return MetropolisWalker(other.state)
