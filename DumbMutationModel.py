import numpy as np
import random

class DumbMutationModel:
    def __init__(self, new=True, w1=None, w2=None, w3=None, b1=None, b2=None, b3=None):
        n = [5, 8, 8, 3]
        self.w1 = np.random.randn(n[1], n[0])
        self.w2 = np.random.randn(n[2], n[1])
        self.w3 = np.random.randn(n[3], n[2])
        self.b1 = np.random.randn(n[1], 1)
        self.b2 = np.random.randn(n[2], 1)
        self.b3 = np.random.randn(n[3], 1)

        if (not new):
            self.w1 = w1
            self.w2 = w2
            self.w3 = w3
            self.b1 = b1
            self.b2 = b2
            self.b3 = b3

    async def feed_forward(self, A0):
        A0 = A0.T
        layer_1_raw = self.w1 @ A0 + self.b1
        layer_1_normalized = await sigmoid(layer_1_raw)

        layer_2_raw = self.w2 @ layer_1_normalized + self.b2
        layer_2_normalized = await sigmoid(layer_2_raw)

        output_raw = self.w3 @ layer_2_normalized + self.b3
        output_normalized = await sigmoid(output_raw)

        return output_normalized

    async def mutate_all_nodes(self, prob):
        self.w1, self.b1 = await mutate_random_nodes(self.w1, self.b1, prob)
        self.w2, self.b2 = await mutate_random_nodes(self.w2, self.b2, prob)
        self.w3, self.b3 = await mutate_random_nodes(self.w3, self.b3, prob)


async def sigmoid(matrix):
        return 1 / (1 + np.exp(-1 * matrix))

async def mutate_random_nodes(weights, biases, chance):
    for index, node in enumerate(weights):
        if (random.random() < chance):
            for weight in node:
                weight += random.uniform(-1, 1)
            biases[index] += random.uniform(-1, 1)
    return weights, biases
