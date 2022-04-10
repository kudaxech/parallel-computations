import random
import copy
import pickle
from mpi4py import MPI
from timeit import default_timer as timer

def split(nums, pivot):
    i, j = 0, len(nums) - 1
    max_index = j
    while i <= j:
        while i <= max_index and nums[i] < pivot: i += 1
        while j >= 0 and nums[j] > pivot: j -= 1
        if i < j:
            nums[i], nums[j] = nums[j], nums[i]
            i, j = i + 1, j - 1
        elif i == j:
            return (nums[:j], nums[i:])
    return (nums[:j+1], nums[i:])

def quicksort(nums, fst, lst):
    if fst >= lst: return
 
    i, j = fst, lst
    pivot = nums[random.randint(fst, lst)]
 
    while i <= j:
        while nums[i] < pivot: i += 1
        while nums[j] > pivot: j -= 1
        if i < j:
            nums[i], nums[j] = nums[j], nums[i]
            i, j = i + 1, j - 1
        elif i == j:
            j-=1
    quicksort(nums, fst, j)
    quicksort(nums, i, lst)

class group_info:
    def __init__(self, rank, group_size, groups_number):
        self.rank = rank
        self.group_size = group_size
        self.group_number = groups_number
    
    def group_id(self) -> int:
        return int(self.rank / self.group_size)

    def leader_id(self) -> int:
        return self.group_size * self.group_id()
    
    def in_the_left_group_part(self) -> bool:
        if self.rank - self.leader_id() < self.group_size / 2:
            return True
        else:
            return False

    def im_leader(self) -> bool:
        if self.rank == self.leader_id():
            return True
        else:
            return False
    
    def cores_in_group(self):
        cores_id = []
        leader_id = self.leader_id()

        for i in range(1, self.group_size):
            cores_id.append(leader_id + i)
        return cores_id

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

groups_number = 1
group_size = copy.copy(size)

for N in range(100000,2100000,100000):
    
    start = MPI.Wtime()

    rand_int = random.randint(0, N - 1)    

    for iter in range(0,800,1):

        if rank == 0:
            with open(f'{N}','rb') as file:
                array = pickle.load(file)
            emergency_element = array[rand_int]
        else:
            emergency_element = None
            array = None

        emergency_element = comm.bcast(emergency_element, root=0)
        array = comm.bcast(array, root=0)
        a_len = len(array)

        if rank == 0:
            array = array[:int(a_len/size)]
        elif rank == size - 1:
            array = array[int(rank * a_len / size):]
        else:
            array = array[int(rank*a_len/size): int((rank + 1) * a_len / size)]

        while groups_number < size:
            core = group_info(rank, group_size, groups_number)
            if core.im_leader():
                if len(array) == 0:
                    pivot = emergency_element
                else:
                    pivot = array[random.randint(0, len(array) - 1)]
                other_cores_id = core.cores_in_group()
                for id in other_cores_id:
                    comm.send(pivot, dest = id)
                left, right = split(array, pivot)
                comm.send(right, dest = core.rank + group_size / 2)
                right = comm.recv(source = core.rank + group_size / 2)

                array = left + right
            else:
                pivot = comm.recv(source = core.leader_id())
                left, right = split(array, pivot)

                if core.in_the_left_group_part():
                    comm.send(right, dest = core.rank + group_size / 2)
                    right = comm.recv(source = core.rank + group_size / 2)
                    array = left + right
                else:
                    min = left
                    left = comm.recv(source = core.rank - group_size / 2)
                    comm.send(min, dest = core.rank - group_size / 2)
                    array = left + right

            groups_number *= 2
            group_size = int(group_size/2)

        # Сюда заходим, когда каждое ядро имеет один массив и группа состоит только из одного ядра

        # сортируем полученный массив
        quicksort(array, 0, len(array) - 1)

        # отправляем полученный массив нулевому ядру

        result = comm.reduce(array, op=MPI.SUM, root=0)

    stop = MPI.Wtime()


    if rank == 0:
        print((stop - start) / 800)