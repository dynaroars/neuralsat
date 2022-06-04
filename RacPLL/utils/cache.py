
class BacksubInstance:

    def __init__(self, layers_mapping, assignment, backsub_dict):
        self.assignment = assignment
        self.backsub_dict = backsub_dict
        self.layers_mapping = layers_mapping
        self.layers_assignment = {
            k: {_: assignment[_] for _ in v if _ in assignment} for k, v in layers_mapping.items()
        }
        self.max_num_hidden = sum([len(layers_mapping[_]) for _ in layers_mapping])

    def get_cache(self, new_assignment):
        cached_nodes = []
        cached_nodes += self.layers_mapping[0]
        for idx in range(len(self.layers_mapping) - 1):
            layer_assignment = {n: new_assignment[n] for n in self.layers_mapping[idx] if n in new_assignment}
            if len(layer_assignment) == len(self.layers_mapping[idx]):
                if layer_assignment == self.layers_assignment[idx]:
                    cached_nodes += list(self.layers_mapping[idx + 1])
                else:
                    break
            else:
                break
        return cached_nodes



class BacksubCacher:

    def __init__(self, layers_mapping, max_caches=10):
        self.layers_mapping = layers_mapping
        self.caches = []
        self.max_caches = max_caches
        

    def put(self, assignment, backsub_dict):
        instance = BacksubInstance(self.layers_mapping, assignment, backsub_dict)
        if self.full():
            self.get()
        self.caches.append(instance)


    def get(self):
        if self.empty():
            return None
        return self.caches.pop(0)


    def full(self):
        return len(self.caches) == self.max_caches


    def empty(self):
        return len(self.caches) == 0


    def get_cache(self, assignment):
        if self.empty():
            return None
        list_caches = []
        for idx, instance in enumerate(self.caches):
            c = instance.get_cache(assignment)
            list_caches.append((idx, c, len(c)))
            if len(c) == instance.max_num_hidden:
                break

        list_caches = sorted(list_caches, key=lambda tup: tup[2], reverse=True)
        idx, cached_nodes, _ = list_caches[0]
        backsub_dict = {n: self.caches[idx].backsub_dict[n] for n in cached_nodes}
        return backsub_dict

    def __len__(self):
        return len(self.caches)

if __name__ == '__main__':
    layers_mapping = {0 : [1, 2, 3], 1: [4, 5], 2: [6, 7, 8]}
    bc = BacksubCacher(layers_mapping, max_caches=3)

    assignment = {1: False, 2: False}
    bc.put(assignment, {})

    assignment = {1: False, 2: False, 3: True, 4: True, 5: True, 6: True, 7: True, 8: True}
    bc.put(assignment, {})


    # assignment = {1: False, 2: False, 3: False, 4: True, 5: False, 6: True, 7: True, 8: True}
    # bc.put(assignment, {})


    new_assignment = {1: False,  2: False, 3: True, 4:True, 5:True}
    c = bc.get_cache(new_assignment)
    print(c)

