import random
from queue import PriorityQueue
from copy import deepcopy  # board states are nested lists


class Board:
    def __init__(self, size=0, state=None) -> None:
        if size:
            self.generate_state(size)
            self.zero_row, self.zero_column = self.find_zero_pos()
            while not self.is_solvable():
                self.generate_state(size)
                self.zero_row, self.zero_column = self.find_zero_pos()
        else:
            self.data = state

        self.zero_row, self.zero_column = self.find_zero_pos()

    def __str__(self) -> str:
        result = '\n'
        for row in self.data:
            result += str(row) + '\n'

        return result[:-1]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int) -> int or list:
        return self.data[item]

    def __eq__(self, other) -> bool:
        return self.data == other.data

    def __lt__(self, other) -> bool:
        return self.total_distance() < other.total_distance()

    def __le__(self, other) -> bool:
        return self.total_distance() <= other.total_distance()

    # Hash function needs .
    #
    def __hash__(self) -> int:
        """
        In Python, dictionary is hash table. So, as we want to use Board
        object as key in dictionaries, object must have hash function.
        hash(object) is called when we use dict[object] call.
        :return: the hash value for the given Board object
        """
        return hash(str(self))

    def total_distance(self):
        return self.manhattan() + self.hamming()

    def generate_state(self, size) -> None:
        self.data = []
        state = list(range(size ** 2))
        random.shuffle(state)

        for i in range(size):
            self.data.append([])
            for j in range(size):
                self.data[i].append(state.pop())

    def hamming(self) -> int:
        result = len(self) ** 2 - 1
        for i in range(len(self) ** 2 - 1):
            if self.data[i // len(self)][i % len(self)] == i + 1:
                result -= 1
        return result

    def manhattan(self) -> int:
        manhattan_distance = 0
        for i in range(len(self)):
            for j in range(len(self)):
                if self.data[i][j]:
                    manhattan_distance += (abs(i - ((self.data[i][j] - 1) // len(self))) +
                                           abs(j - ((self.data[i][j] - 1) % len(self))))
        return manhattan_distance

    def is_goal(self) -> bool:
        return True if not self.hamming() else False

    def create_goal(self) -> object:
        result = []
        data = list(range(1, len(self) ** 2))
        data.append(0)
        data = data[::-1]

        for i in range(len(self)):
            result.append([])
            for j in range(len(self)):
                result[i].append(data.pop())

        return Board(0, result)

    def is_solvable(self) -> bool:
        inversions = 0

        if self.is_goal():
            return True

        for i in range(len(self)):
            for j in range(len(self)):
                if self.data[i][j]:
                    for j1 in range(0, j):
                        inversions += self.data[i][j1] > self.data[i][j]

                    for i1 in range(i):
                        for j1 in range(len(self)):
                            inversions += self.data[i1][j1] > self.data[i][j]

        if len(self) % 2:
            return True if (not inversions % 2) else False

        else:
            if inversions % 2:
                return True if (not self.zero_row % 2) else False

            return True if (self.zero_row % 2) else False

    def find_zero_pos(self):
        for i in range(len(self)):
            for j in range(len(self)):
                if self.data[i][j] == 0:
                    return i, j

    def get_children(self):
        children = []

        has_child_at_left = False if self.zero_column == 0 else True
        has_child_at_right = False if self.zero_column == len(self) - 1 else True
        has_child_at_top = False if self.zero_row == 0 else True
        has_child_at_bottom = False if self.zero_row == len(self) - 1 else True

        if has_child_at_left:
            data = deepcopy(self.data)
            data[self.zero_row][self.zero_column], data[self.zero_row][self.zero_column-1] =\
                data[self.zero_row][self.zero_column-1], data[self.zero_row][self.zero_column]
            children.append(Board(0, data))
        if has_child_at_right:
            data = deepcopy(self.data)
            data[self.zero_row][self.zero_column], data[self.zero_row][self.zero_column+1] =\
                data[self.zero_row][self.zero_column+1], data[self.zero_row][self.zero_column]
            children.append(Board(0, data))
        if has_child_at_top:
            data = deepcopy(self.data)
            data[self.zero_row][self.zero_column], data[self.zero_row-1][self.zero_column] =\
                data[self.zero_row-1][self.zero_column], data[self.zero_row][self.zero_column]
            children.append(Board(0, data))
        if has_child_at_bottom:
            data = deepcopy(self.data)
            data[self.zero_row][self.zero_column], data[self.zero_row+1][self.zero_column] =\
                data[self.zero_row+1][self.zero_column], data[self.zero_row][self.zero_column]
            children.append(Board(0, data))

        return children


class Solver:
    def __init__(self, board: Board):
        self.board = board
        self.goal = board.create_goal()
        self.solution = self.solve()

    def __next__(self):
        pass

    def moves(self) -> int:
        return len(self.solution)

    def solve(self) -> list[Board]:
        if not self.board.is_solvable():
            print("CAN'T SOLVE")
            return []

        frontier = PriorityQueue()
        frontier.put((self.board, 0))

        came_from = dict()
        came_from[self.board] = None
        cost_so_far = dict()
        cost_so_far[self.board] = 0

        while not frontier.empty():
            current = frontier.get()[0]

            if current.is_goal():
                solution = []

                flag = 1
                while flag != 0:
                    solution.append(current)
                    current = came_from[current]

                    if came_from[current] is None:
                        flag = 0

                return solution[::-1]

            for child in current.get_children():
                priority = cost_so_far[current] + child.total_distance()
                if child not in cost_so_far or priority < cost_so_far[child]:
                    cost_so_far[child] = priority
                    frontier.put((child, priority))
                    came_from[child] = current


def main():
    b = Board(3)
    s = Solver(b)
    print(s.moves())


if __name__ == '__main__':
    main()
