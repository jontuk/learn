import os


def read_state_actions():
    state = None
    rewards = dict()

    states = dict()

    for line in open(os.path.join('..', 'logs', 'sim_improved-learning.txt')).readlines():
        if line[0] == '(':
            state = eval(line)
        elif line.startswith(' -- '):
            spl = line.split(' : ')
            action = spl[0][4:]
            rewards[None if action == 'None' else action] = float(spl[1])
        else:
            if rewards:
                states[state] = rewards
            rewards = dict()
    return states


def is_match(policy_state, state):
    for i, val in enumerate(policy_state):
        if val != '*' and state[i] != val:
            return False
    return True


def check_optimal_policy(state_actions):
    matched_states = 0
    correct_actions = 0

    optimal_policies = list()

    # Right turn, left going forward, stay
    optimal_policies += [(('forward', '*', '*', 'red', 'right'), None)]

    # Right turn, other cases, go
    optimal_policies += [(('*', '*', '*', 'red', 'right'), 'right')]

    # Any other red light
    optimal_policies += [(('*', '*', '*', 'red', '*'), None)]

    for state in state_actions:
        for policy_state, policy_action in optimal_policies:
            if is_match(policy_state, state): # and state_actions[state][policy_action] != 0:
                best_action = sorted([(reward, action) for action, reward in state_actions[state].items()])[-1][1]
                if best_action == policy_action:
                    correct_actions += 1
                matched_states += 1
                print('State: %s Optimal action: %s Q: %s Agent optimal? %s'
                      % (state, policy_action, state_actions[state], best_action == policy_action))
                break
    print('Optimal policy followed in %s/%s matched states' % (correct_actions, matched_states))


if '__main__' == __name__:
    state_actions = read_state_actions()
    check_optimal_policy(state_actions)