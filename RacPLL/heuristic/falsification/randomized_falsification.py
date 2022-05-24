import random
import torch
import time

import settings

class RandomizedFalsification:

    def __init__(self, net, spec):

        self.bounds = spec.bounds
        self.mat = spec.mat

        self.net = net

        self.n_runs = 50
        self.n_samples = 100

        self.n_pos_samples = 10

        self._find_target_and_direction()

        print('target:', self.targets)
        print('direction:', self.directions)

        if settings.SEED is not None:
            random.seed(settings.SEED)


    def _find_target_and_direction(self):
        self.targets = []
        self.directions = []

        for arr, _ in self.mat:
            target_dict = dict.fromkeys(range(self.net.n_output), 0)
            obj_dict = dict.fromkeys(range(self.net.n_output), 0)
            for k in range(len(arr)):
                for kk in range(len(arr[k])):
                    if (arr[k][kk] != 0):
                        cntr = target_dict[kk] + 1
                        target_dict[kk] = cntr
                    if (arr[k][kk] < 0):
                        cntr = obj_dict[kk] + 1
                        obj_dict[kk] = cntr
            target = max(target_dict, key=target_dict.get)
            obj_type = max(obj_dict, key=obj_dict.get)
            if (target == obj_type):
                direction = 'maximization'
            else:
                direction = 'minimization'

            self.targets.append(target)
            self.directions.append(direction)


    def eval_constraints(self, input_ranges=None, constraints=None):
        if input_ranges is None:
            input_ranges = torch.tensor(self.bounds, dtype=settings.DTYPE)
        input_ranges_clone = input_ranges.clone()

        for target, direction in zip(self.targets, self.directions):
            stat, adv = self._sampling(input_ranges, self.mat, target, direction)
            if stat == 'violated':
                return stat, adv
            input_ranges = input_ranges_clone
        return 'unknown', None


    def eval(self, input_ranges=None, timeout=1):
        start = time.time()
        count = 0
        while True:
            count += 1
            tic = time.time()
            stat, adv = self.eval_constraints(input_ranges)
            # print(count, time.time() - tic)
            if stat == 'violated':
                return stat, adv

            time_elapsed = time.time() - start
            if time_elapsed > timeout:
                return 'timeout', None


    def _sampling(self, input_ranges, output_props, target, direction):
        old_pos_samples = []

        for _ in range(self.n_runs):
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
                if input_ranges[i][1] - input_ranges[i][0] > 1e-6:
                   flag = True
                   break
            if not flag:
                return 'unknown', None

            input_ranges = self._learning(pos_samples, neg_samples, input_ranges)

        return 'unknown', None


    def _learning(self, pos_samples, neg_samples, input_ranges):
        new_input_ranges = input_ranges.clone()
        random.shuffle(neg_samples)

        for neg_sample in neg_samples:
            pos_sample = random.choice(pos_samples)
            dim = random.randint(0, int(self.net.n_input) - 1)

            # for dim in range(self.net.n_input):
            pos_val = pos_sample[0][dim]
            neg_val = neg_sample[0][dim]
            if pos_val > neg_val:
                temp = torch.round(random.uniform(neg_val, pos_val), decimals=6).to(settings.DTYPE)
                if temp <= new_input_ranges[dim][1] and temp >= new_input_ranges[dim][0]:
                    new_input_ranges[dim][0] = temp
            else:
                temp = torch.round(random.uniform(pos_val, neg_val), decimals=6).to(settings.DTYPE)
                if temp <= new_input_ranges[dim][1] and temp >= new_input_ranges[dim][0]:
                    new_input_ranges[dim][1] = temp
        return new_input_ranges


    def _split_samples(self, samples, target, direction):
        if direction == 'minimization':
            sorted_samples = sorted(samples, key=lambda tup: tup[1][target])
        else:
            sorted_samples = sorted(samples, key=lambda tup: tup[1][target], reverse=True)

        pos_samples = sorted_samples[:self.n_pos_samples]
        neg_samples = sorted_samples[self.n_pos_samples:]

        return pos_samples, neg_samples


    def _make_samples(self, input_ranges, output_props):
        samples = []
        for _ in range(self.n_samples):
            s_in = torch.Tensor([
                torch.round(random.uniform(input_ranges[i][0], input_ranges[i][1]), decimals=6)
                for i in range(self.net.n_input)]).to(settings.DTYPE)
            s_in = s_in.view(self.net.input_shape)
            s_out = self.net(s_in)
            stat = self._check_property(output_props, s_out)
            if stat == 'violated':
                return stat, (s_in, s_out)
            samples.append((s_in.view(-1), s_out.view(-1)))
        return stat, samples
        

    def _check_property(self, output_props, output):
        for prop_mat, prop_rhs in output_props:
            prop_mat = torch.tensor(prop_mat, dtype=settings.DTYPE)
            prop_rhs = torch.tensor(prop_rhs, dtype=settings.DTYPE)
            vec = prop_mat @ output.squeeze(0)
            if torch.all(vec <= prop_rhs):
                return 'violated'
        return 'unknown'
