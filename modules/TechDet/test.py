import unittest

import torch
import torch.nn as nn

from squeezeformer.encoder import SqueezeformerLayers


class TestSqueezeformer(unittest.TestCase):
    def test_forward(self):
        BATCH_SIZE = 4
        SEQ_LENGTH = 500
        INPUT_SIZE = 512

        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        model = SqueezeformerLayers().to(device)

        inputs = torch.FloatTensor(BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE).to(device)

        outputs = model(inputs)

    # def test_backward(self):
    #     BATCH_SIZE = 4
    #     SEQ_LENGTH = 500
    #     INPUT_SIZE = 80

    #     device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    #     criterion = nn.CTCLoss().to(device)
    #     model = SqueezeformerLayers().to(device)

    #     inputs = torch.FloatTensor(BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE).to(device)
    #     input_lengths = torch.IntTensor([500, 450, 400, 350]).to(device)
    #     targets = torch.LongTensor(
    #         [
    #             [1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
    #             [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
    #             [1, 3, 3, 3, 3, 3, 4, 2, 0, 0],
    #             [1, 3, 3, 3, 3, 3, 6, 2, 0, 0],
    #         ]
    #     ).to(device)
    #     target_lengths = torch.LongTensor([9, 8, 7, 7]).to(device)

    #     for _ in range(3):
    #         outputs, output_lengths = model(inputs, input_lengths)
    #         loss = criterion(outputs.transpose(0, 1), targets[:, 1:], output_lengths, target_lengths)
    #         loss.backward()


if __name__ == "__main__":
    unittest.main()