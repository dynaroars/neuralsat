import random
import torch
import time

class FastFalsify:

    def __init__(self, net, spec_list):

        self.n_inputs = net.input_shape[1]
        self.n_outputs = net.output_shape[1]
        
        self.spec_list = spec_list
        self.net = net

        self.n_runs = 10
        self.n_samples = 5

        self._find_target_and_direction()


    def _find_target_and_direction(self):
        target_dict = dict.fromkeys(range(self.n_outputs), 0)
        obj_dict = dict.fromkeys(range(self.n_outputs), 0)

        for spec in self.spec_list[0][1]:
            arr = spec[0]
            for k in range(len(arr)):
                for kk in range(len(arr[k])):
                    if (arr[k][kk] != 0):
                        cntr = target_dict[kk] + 1
                        target_dict[kk] = cntr
                    if (arr[k][kk] < 0):
                        cntr = obj_dict[kk] + 1
                        obj_dict[kk] = cntr
        self.target = max(target_dict, key=target_dict.get)
        obj_type = max(obj_dict, key=obj_dict.get)

        if (self.target == obj_type):
            self.direction = 'maximization'
        else:
            self.direction = 'minimization'


    def eval(self, timeout=2):
        start = time.time()
        while True:
            stat, adv = self._evaluate()
            if stat == 'violated':
                return stat, adv

            time_elapsed = time.time() - start
            if time_elapsed > timeout:
                return 'timeout', None


    def _evaluate(self):
        for spec in self.spec_list:
            input_ranges = torch.tensor(spec[0], dtype=torch.float32)
            output_props = spec[1]
            random.seed(0)
            
            stat, adv = self._sampling(input_ranges, output_props)
            if stat == 'violated':
                return stat, adv
        return 'unknown', None


    def _sampling(self, input_ranges, output_props):
        old_pos_samples = []

        for _ in range(self.n_runs):
            stat, samples = self._make_samples(input_ranges, output_props)
            if stat == 'violated':
                return stat, samples
            pos_samples, neg_samples = self._segregate_samples(samples, old_pos_samples)
            old_pos_samples = pos_samples

            flag = False
            for i in range(self.n_inputs):
                if input_ranges[i][1] - input_ranges[i][0] > 1e-6:
                   flag = True
                   break
            if not flag:
                return 'unknown', None

            self._learning(pos_samples, neg_samples, input_ranges)

        return 'unknown', None


    def _learning(self, pos_samples, neg_samples, input_ranges):
        for i in range(len(neg_samples)):
            dim = random.randint(0, int(self.n_inputs) - 1)
            pos_val = pos_samples[0][0][dim]
            neg_val = neg_samples[i][0][dim]
            if pos_val > neg_val:
                temp = torch.round(random.uniform(neg_val, pos_val), decimals=6)
                if temp <= input_ranges[dim][1] and temp >= input_ranges[dim][0]:
                    input_ranges[dim][0] = temp
            else:
                temp = torch.round(random.uniform(pos_val, neg_val), decimals=6)
                if temp <= input_ranges[dim][1] and temp >= input_ranges[dim][0]:
                   input_ranges[dim][1] = temp


    def _segregate_samples(self, samples, old_pos_samples):
        pos_samples = []
        neg_samples = []

        if self.direction == 'maximization':
            s_out = samples[0][1]
            large = s_out[self.target]
            last_idx = 0
            pos_samples.append(samples[0])
            for i in range(1, len(samples)):
                s_out = samples[i][1]
                new_large = s_out[self.target]
                if new_large > large :
                    pos_samples.remove(samples[last_idx])
                    pos_samples.append(samples[i])
                    neg_samples.append(samples[last_idx])
                    last_idx = i
                    large = new_large
                else:
                    if new_large < large :
                       neg_samples.append(samples[i])

            if len(old_pos_samples) > 0:
                cur_pos_samples1 = pos_samples[0][1]
                old_pos_samples1 = old_pos_samples[0][1]
                if old_pos_samples1[self.target] > cur_pos_samples1[self.target]:
                    neg_samples.append(pos_samples[0])
                    pos_samples.remove(pos_samples[0])
                    pos_samples.append(old_pos_samples[0])

        elif self.direction == 'minimization':
            s_out = samples[0][1]
            small = s_out[self.target]
            last_idx = 0
            pos_samples.append(samples[0])
            for i in range(1, len(samples)):
                s_out = samples[i][1]
                new_small = s_out[self.target]
                if new_small < small:
                    pos_samples.remove(samples[last_idx])
                    pos_samples.append(samples[i])
                    neg_samples.append(samples[last_idx])
                    last_idx = i
                    small = new_small
                else:
                    if new_small < small:
                       neg_samples.append(samples[i])

            if len(old_pos_samples) > 0:
                cur_pos_samples1 = pos_samples[0][1]
                old_pos_samples1 = old_pos_samples[0][1]
                if old_pos_samples1[self.target] < cur_pos_samples1[self.target]:
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
                for i in range(self.n_inputs)])
            s_out = self.net(s_in)
            stat = self._check_property(output_props, s_out)
            if stat == 'violated':
                return stat, (s_in, s_out[0])
            samples.append((s_in, s_out[0]))

        return stat, samples

    def _check_property(self, output_props, output):
        for prop_mat, prop_rhs in output_props:
            prop_mat = torch.tensor(prop_mat, dtype=torch.float32)
            prop_rhs = torch.tensor(prop_rhs, dtype=torch.float32)
            vec = prop_mat @ output.transpose(0, 1)
            if torch.all(vec <= prop_rhs):
                return 'violated'
        return 'unknown'
