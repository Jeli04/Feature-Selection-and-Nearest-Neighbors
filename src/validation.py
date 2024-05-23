import numpy as np

class Validator:
    def __init__(self, filename, test_size=0.2) -> None:
        with open(filename, 'r') as file:
            lines = file.readlines()

        self.data = np.array([list(map(float, line.split())) for line in lines])
        self.train = self.data[:int(len(self.data) * 0.8)]
        self.test = self.data[int(len(self.data) * 0.8):]

    def k_fold(self, k):
        return 

    def eval(self):
        return


test = Validator("data/large-test-dataset.txt")
print(test.data)
