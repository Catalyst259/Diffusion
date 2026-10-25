"""Ensure an EDM-to-backbone noise encoding scale is consistent end to end."""
import unittest
import torch
from torch import nn
from DiTLab.algorithms.edm import EDMAlgorithm


class RecordingNetwork(nn.Module):
    def forward_denoise(self, images, labels, noise_condition, label_mask=None):
        self.condition = noise_condition.detach().clone()
        return torch.zeros_like(images)


class ConditionScaleTests(unittest.TestCase):
    def test_training_and_denoising_encode_the_same_known_sigma(self):
        sigma = 0.3
        model = RecordingNetwork()
        method = EDMAlgorithm(noise_condition_scale=100.0, p_mean=float(torch.tensor(sigma).log()),
                              p_std=0.0, label_dropout=0.0)
        images = torch.zeros(2,1,4,4)
        labels = torch.tensor([0,9])
        method.loss(model,images,labels,generator=torch.Generator().manual_seed(2))
        expected = torch.full((2,), float(torch.tensor(sigma).log()/4)*100)
        torch.testing.assert_close(model.condition,expected)
        method._denoise(model,images,labels,torch.tensor(sigma))
        torch.testing.assert_close(model.condition,expected)

    def test_original_conditioning_remains_default_and_scale_is_validated(self):
        self.assertEqual(EDMAlgorithm().noise_condition_scale, 1.0)
        for invalid in (0,-1,float('nan'),float('inf')):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                EDMAlgorithm(noise_condition_scale=invalid)


if __name__ == '__main__':
    unittest.main()
