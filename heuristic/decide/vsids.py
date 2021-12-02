
class PriorityQueue:

    def __init__(self, start_list):
        # The first element is removed as it is not related  to any variable or literal
        self.size = len(start_list) - 1
        temp = start_list[1:]

        # Max heap with respect to the priority scores
        self.heap = []

        # Array that maps elements to their indices in the heap
        self.indices = []

        for idx, x in enumerate(temp):
            self.heap.append([x, idx+1])
            self.indices.append(idx)

        # Convert an array into heap
        for i in range(int(self.size/2)-1, -1, -1):
            self.heapify(i)

    def swap(self, idx1, idx2):
        # Swap the nodes in the heap array
        temp = self.heap[idx1]
        self.heap[idx1] = self.heap[idx2]
        self.heap[idx2] = temp

        # Swap the indices of the nodes in the indices array 
        p1 = self.heap[idx1][1]
        p1 -= 1
        p2 = self.heap[idx2][1]
        p2 -= 1
        temp = self.indices[p1]
        self.indices[p1] = self.indices[p2]
        self.indices[p2] = temp

    def heapify(self, node_index):
        # Calculate the max priority between the node
        maxp = self.heap[node_index][0]

        left_index = 2*node_index + 1
        if left_index < self.size:
            pr = self.heap[left_index][0]
            if pr > maxp:
                maxp = pr

        right_index = 2*node_index + 2
        if right_index < self.size:
            pr = self.heap[right_index][0]
            if pr > maxp:
                maxp = pr

        # If max priority is not of root
        if maxp != self.heap[node_index][0]:
            if left_index < self.size and maxp == self.heap[left_index][0]:
                self.swap(left_index, node_index)
                self.heapify(left_index)
            else:
                self.swap(right_index, node_index)
                self.heapify(right_index)

    def remove(self, key):
        if self.indices[key-1] == -1:
            return

        # Replace the node to be deleted with the final node
        pos = self.indices[key-1]
        this_node_pr = self.heap[pos][0]
        final_node_pr = self.heap[self.size-1][0]
        self.swap(pos, self.size-1)
        self.size -= 1
        self.indices[key-1] = -1

        if final_node_pr > this_node_pr:
            # If the replaced node has a higher priority, then the deleted node
            # Traverse the heap upwards and replace the node with its parent until its parent is bigger than it.
            par = pos
            while par != 0:
                temp = par
                par = int((par - 1)/2)
                if self.heap[temp][0] > self.heap[par][0]:
                    self.swap(temp, par)
                else:
                    break
        elif this_node_pr > final_node_pr:
            # If replaced node has a lower priority than the removed node
            # then use heapify to maintain the heap structure
            self.heapify(pos)
