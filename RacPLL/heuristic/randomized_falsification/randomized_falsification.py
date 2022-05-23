import random
import torch
import time

import settings

class RandomizedFalsification:

    def __init__(self, net, spec):

        self.bounds = spec.bounds
        self.mat = spec.mat

        self.net = net

        self.n_runs = 10
        self.n_samples = 10

        self._find_target_and_direction()

        # print('target:', self.targets)
        # print('direction:', self.directions)
        # exit()


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

            # pos_samples, neg_samples = [], []
            # for target, direction in zip(self.targets, self.directions):
            pos_samples, neg_samples = self._segregate_samples(samples, old_pos_samples, target, direction)
                # pos_samples += p
                # neg_samples += n
            # print(len(pos_samples + neg_samples))

            old_pos_samples = pos_samples

            flag = False
            for i in range(self.net.n_input):
                if input_ranges[i][1] - input_ranges[i][0] > 1e-6:
                   flag = True
                   break
            if not flag:
                return 'unknown', None

            self._learning(pos_samples, neg_samples, input_ranges)

        return 'unknown', None


    def _learning(self, pos_samples, neg_samples, input_ranges):
        for i in range(len(neg_samples)):
            dim = random.randint(0, int(self.net.n_input) - 1)
            pos_val = pos_samples[0][0][dim]
            neg_val = neg_samples[i][0][dim]
            if pos_val > neg_val:
                temp = torch.round(random.uniform(neg_val, pos_val), decimals=6).to(settings.DTYPE)
                if temp <= input_ranges[dim][1] and temp >= input_ranges[dim][0]:
                    input_ranges[dim][0] = temp
            else:
                temp = torch.round(random.uniform(pos_val, neg_val), decimals=6).to(settings.DTYPE)
                if temp <= input_ranges[dim][1] and temp >= input_ranges[dim][0]:
                   input_ranges[dim][1] = temp


    def _segregate_samples(self, samples, old_pos_samples, target, direction):
        pos_samples = []
        neg_samples = []

        if direction == 'maximization':
            s_out = samples[0][1]
            large = s_out[target]
            last_idx = 0
            pos_samples.append(samples[0])
            for i in range(1, len(samples)):
                s_out = samples[i][1]
                new_large = s_out[target]
                if new_large > large:
                    pos_samples.remove(samples[last_idx])
                    pos_samples.append(samples[i])
                    neg_samples.append(samples[last_idx])
                    last_idx = i
                    large = new_large
                else:
                    neg_samples.append(samples[i])

            if len(old_pos_samples) > 0:
                cur_pos_samples1 = pos_samples[0][1]
                old_pos_samples1 = old_pos_samples[0][1]
                if old_pos_samples1[target] > cur_pos_samples1[target]:
                    neg_samples.append(pos_samples[0])
                    pos_samples.remove(pos_samples[0])
                    pos_samples.append(old_pos_samples[0])

        elif direction == 'minimization':
            s_out = samples[0][1]
            small = s_out[target]
            last_idx = 0
            pos_samples.append(samples[0])
            for i in range(1, len(samples)):
                s_out = samples[i][1]
                new_small = s_out[target]
                if new_small < small:
                    pos_samples.remove(samples[last_idx])
                    pos_samples.append(samples[i])
                    neg_samples.append(samples[last_idx])
                    last_idx = i
                    small = new_small
                else:
                    neg_samples.append(samples[i])

            if len(old_pos_samples) > 0:
                cur_pos_samples1 = pos_samples[0][1]
                old_pos_samples1 = old_pos_samples[0][1]
                if old_pos_samples1[target] < cur_pos_samples1[target]:
                    neg_samples.append(pos_samples[0])
                    pos_samples.remove(pos_samples[0])
                    pos_samples.append(old_pos_samples[0])

        else:
            raise NotImplementedError

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
