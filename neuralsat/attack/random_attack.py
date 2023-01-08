import random
import torch
import time

from attack.error import AttackTimeoutError
import arguments

class RandomAttack:

    def __init__(self, net, spec, timeout=0.2, seed=None):

        self.bounds = spec.bounds
        self.mat = spec.mat

        self.net = net
        self.device = net.device

        self.n_runs = 10
        self.n_samples = 50

        self.n_pos_samples = 5

        self.dtype = arguments.Config['dtype']

        self._find_target_and_direction()
        # print(self.targets)
        # print(self.directions)

        if seed is not None:
            self.manual_seed(seed)

        self.timeout = timeout

        self.seed = seed

    def manual_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        self.seed = seed

    def _find_target_and_direction(self):
        self.targets = []
        self.directions = []

        for arr, _ in self.mat:
            target_dict = dict.fromkeys(range(self.net.n_output), 0)
            obj_dict = dict.fromkeys(range(self.net.n_output), 0)
            for k in range(len(arr)):
                for kk in range(len(arr[k])):
                    if (arr[k][kk] != 0):
                        target_dict[kk] += 1
                    if (arr[k][kk] < 0):
                        obj_dict[kk] += 1

            target = max(target_dict, key=target_dict.get)
            obj_type = max(obj_dict, key=obj_dict.get)

            if (target == obj_type):
                direction = 'maximize'
            else:
                direction = 'minimize'

            if target not in self.targets:
                self.targets.append(target)
                self.directions.append(direction)

            target = sorted(target_dict.items(), key=lambda item: item[1])[-1][0]
            if target not in self.targets:
                obj_type = sorted(obj_dict.items(), key=lambda item: item[1])[-1][0]
                if (target == obj_type):
                    direction = 'maximize'
                else:
                    direction = 'minimize'
                self.targets.append(target)
                self.directions.append(direction)



    def eval_constraints(self, input_ranges=None, constraints=None):
        if input_ranges is None:
            input_ranges = torch.tensor(self.bounds, dtype=self.dtype, device=self.device)
        input_ranges_clone = input_ranges.clone()

        for target, direction in zip(self.targets, self.directions):
            stat, adv = self._sampling(input_ranges, self.mat, target, direction)
            if stat == 'violated':
                return stat, adv
            input_ranges = input_ranges_clone
        return 'unknown', None


    def run(self, input_ranges=None):
        self.start = time.time()
        count = 0
        while True:
            try:
                count += 1
                stat, adv = self.eval_constraints(input_ranges)
                if stat == 'violated':
                    return True, adv[0]
                self.check_timeout()
            except AttackTimeoutError:
                return False, None

    def check_timeout(self):
        time_elapsed = time.time() - self.start
        if time_elapsed > self.timeout:
            raise AttackTimeoutError

    def _sampling(self, input_ranges, output_props, target, direction):
        # print('sampling')
        old_pos_samples = []

        for _ in range(self.n_runs):
            self.check_timeout()
            stat, samples = self._make_samples(input_ranges, output_props)
            if stat == 'violated':
                return stat, samples

            pos_samples, neg_samples = self._split_samples(samples, target, direction)

            if len(old_pos_samples) > 0:
                pos_samples, neg_samples_2 = self._split_samples(pos_samples + old_pos_samples, target, direction)
                neg_samples = neg_samples_2 + neg_samples

            old_pos_samples = pos_samples

            flag = False
            for i in range(self.net.n_input):
                if input_ranges[i][1] - input_ranges[i][0] > 1e-8:
                   flag = True
                   break
            if not flag:
                return 'unknown', None

            input_ranges = self._learning(pos_samples, neg_samples, input_ranges)

        return 'unknown', None


    def _learning(self, pos_samples, neg_samples, input_ranges):
        # print('learning')
        new_input_ranges = input_ranges.clone()
        random.shuffle(neg_samples)

        pos_sample = random.choice(pos_samples)
        dim = random.randint(0, int(self.net.n_input) - 1)
        for neg_sample in neg_samples:
            self.check_timeout()
            # for dim in range(self.net.n_input):
            pos_val = pos_sample[0][dim]
            neg_val = neg_sample[0][dim]
            if pos_val > neg_val:
                temp = random.uniform(neg_val, pos_val)
                if temp <= new_input_ranges[dim][1] and temp >= new_input_ranges[dim][0]:
                    new_input_ranges[dim][0] = temp
            else:
                temp = random.uniform(pos_val, neg_val)
                if temp <= new_input_ranges[dim][1] and temp >= new_input_ranges[dim][0]:
                    new_input_ranges[dim][1] = temp
        return new_input_ranges


    def _split_samples(self, samples, target, direction):
        # print('split')
        if direction == 'minimize':
            sorted_samples = sorted(samples, key=lambda tup: tup[1][target])
        else:
            sorted_samples = sorted(samples, key=lambda tup: tup[1][target], reverse=True)

        pos_samples = sorted_samples[:self.n_pos_samples]
        neg_samples = sorted_samples[self.n_pos_samples:]

        return pos_samples, neg_samples


    def _make_samples(self, input_ranges, output_props):
        # print('make samples')
        l = input_ranges[:, 0].reshape(self.net.input_shape)
        u = input_ranges[:, 1].reshape(self.net.input_shape)
        s_in = (u-l) * torch.rand(self.n_samples, *self.net.input_shape[1:]).to(self.device) + l
        assert torch.all(s_in >= l) and torch.all(s_in <= u)
        s_out = self.net(s_in)

        self.check_timeout()

        samples = []
        for prop_mat, prop_rhs in output_props:
            prop_mat = torch.tensor(prop_mat, dtype=self.dtype, device=self.device)
            prop_rhs = torch.tensor(prop_rhs, dtype=self.dtype, device=self.device)
            vec = prop_mat @ s_out.transpose(0, 1)
            for i in range(self.n_samples):
                sample = s_in[i].view(-1), s_out[i].view(-1)
                if torch.all(vec[:, i] <= prop_rhs):
                    return 'violated', sample
                samples.append(sample)
        return 'unknown', samples


    def __str__(self):
        return f'RandomAttack(seed={self.seed})'