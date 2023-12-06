import numpy as np
import random
import torch
import time

class RandomAttacker:

    def __init__(self, net, objective, input_shape, device):
        self.net = net
        self.objective = objective
        self.input_shape = input_shape
        self.device = device

        if np.prod(self.input_shape) >= 200:
            return None
        
        self.n_runs = 10
        self.n_samples = 50
        self.n_pos_samples = int(self.n_samples * 0.1)

        self.output_shape = self.net(torch.zeros(self.input_shape, device=device)).shape
        self.target, self.direction = self.get_target_and_direction()

        # print(self.target)
        # print(self.direction)


    def manual_seed(self, seed):
        self.seed = seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        
    def get_target_and_direction(self):
        target_dict = dict.fromkeys(range(np.prod(self.output_shape)), 0)
        obj_dict = dict.fromkeys(range(np.prod(self.output_shape)), 0)
        for arr in self.objective.cs:
            for k in range(len(arr)):
                for kk in range(len(arr[k])):
                    if (arr[k][kk] != 0):
                        target_dict[kk] += 1
                    if (arr[k][kk] < 0):
                        obj_dict[kk] += 1

        target = sorted(target_dict.items(), key=lambda item: item[1])[-1][0]
        obj_type = sorted(obj_dict.items(), key=lambda item: item[1])[-1][0]
        if target == obj_type:
            direction = 'maximize'
        else:
            direction = 'minimize'
        
        return target, direction


    def _attack(self, input_lowers, input_uppers):
        adv = self._sampling(
            input_lowers=input_lowers, 
            input_uppers=input_uppers, 
            target=self.target, 
            direction=self.direction,
        )
        return adv


    def run(self, timeout=1.0):
        if np.prod(self.input_shape) >= 200:
            return False, None
        
        indexes = torch.randperm(self.objective.lower_bounds.shape[0])
        shuffled_lower_bounds = self.objective.lower_bounds[indexes]
        shuffled_upper_bounds = self.objective.upper_bounds[indexes]
        if not torch.equal(self.objective.lower_bounds, shuffled_lower_bounds) or \
            not torch.equal(self.objective.upper_bounds, shuffled_upper_bounds):
                return False, None
            
        input_lowers = self.objective.lower_bounds[0].clone().view(self.input_shape).to(self.device)
        input_uppers = self.objective.upper_bounds[0].clone().view(self.input_shape).to(self.device)

        start = time.time()
        while True:
            adv = self._attack(input_lowers=input_lowers, input_uppers=input_uppers)
            if adv is not None:
                return True, adv
            
            if time.time() - start > timeout:
                return False, None


    def _sampling(self, input_lowers, input_uppers, target, direction):
        old_pos_samples = []

        for it in range(self.n_runs):
            stat, samples = self._make_samples(
                input_lowers=input_lowers, 
                input_uppers=input_uppers,
            )
            if stat:
                return samples

            pos_samples, neg_samples = self._split_samples(
                samples=samples, 
                target=target, 
                direction=direction,
            )

            if len(old_pos_samples) > 0:
                pos_samples, neg_samples_2 = self._split_samples(
                    samples=pos_samples + old_pos_samples, 
                    target=target, 
                    direction=direction,
                )
                neg_samples = neg_samples_2 + neg_samples

            old_pos_samples = pos_samples

            if torch.all((input_uppers - input_lowers) <= 1e-6):
                return None

            input_lowers, input_uppers = self._learning(
                input_lowers=input_lowers, 
                input_uppers=input_uppers, 
                pos_samples=pos_samples, 
                neg_samples=neg_samples,
            )

        return None


    def _learning(self, input_lowers, input_uppers, pos_samples, neg_samples):
        new_input_lowers = input_lowers.clone().flatten()
        new_input_uppers = input_uppers.clone().flatten()
        random.shuffle(neg_samples)

        pos_sample = random.choice(pos_samples)
        dim = random.randint(0, new_input_uppers.numel() - 1)
        for neg_sample in neg_samples:
            pos_val = pos_sample[0][dim]
            neg_val = neg_sample[0][dim]
            if pos_val > neg_val:
                temp = random.uniform(neg_val, pos_val)
                if new_input_lowers[dim] <= temp <= new_input_uppers[dim]:
                    new_input_lowers[dim] = temp
            else:
                temp = random.uniform(pos_val, neg_val)
                if new_input_lowers[dim] <= temp <= new_input_uppers[dim]:
                    new_input_uppers[dim] = temp
                    
        return new_input_lowers.view(self.input_shape), new_input_uppers.view(self.input_shape)


    def _split_samples(self, samples, target, direction):
        if direction == 'minimize':
            sorted_samples = sorted(samples, key=lambda tup: tup[1][target])
        else:
            sorted_samples = sorted(samples, key=lambda tup: tup[1][target], reverse=True)

        pos_samples = sorted_samples[:self.n_pos_samples]
        neg_samples = sorted_samples[self.n_pos_samples:]

        return pos_samples, neg_samples


    def _make_samples(self, input_lowers, input_uppers):
        s_in = (input_uppers - input_lowers) * torch.rand(self.n_samples, *self.input_shape[1:], device=self.device) + input_lowers
        assert torch.all(input_lowers <= s_in) and torch.all(s_in <= input_uppers)
        s_out = self.net(s_in)
        samples = []
        for cs, rhs in zip(self.objective.cs.to(self.device), self.objective.rhs.to(self.device)):
            vec = cs @ s_out.transpose(0, 1)
            for i in range(self.n_samples):
                if torch.all(vec[:, i] <= rhs):
                    return True, s_in[i].view(self.input_shape)
                sample = s_in[i].flatten(), s_out[i].flatten()
                samples.append(sample)
        return False, samples


    def __repr__(self):
        return f'RandomAttack(seed={self.seed}, device={self.device})'
