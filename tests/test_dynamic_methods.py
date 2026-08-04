import unittest
from types import SimpleNamespace

import torch

from backbone.generic_mil import GenericMILBackbone
from models.agem import AGem
from models.conslide import ConSlide
from models.derpp import Derpp
from models.er_ace import ErACE
from models.ewc_on import EwcOn
from models.gdumb import GDumb
from models.joint import Joint
from models.lwf import Lwf
from models.sgd import Sgd
from utils.buffer import Instance_Buffer
from utils.training import evaluate


CLASS_COUNTS = [4, 2, 3, 2, 2, 2, 2, 3, 2, 5]
OFFSETS = [0, 4, 6, 9, 11, 13, 15, 17, 20, 22]


def _args():
    return SimpleNamespace(
        lr=1e-3,
        buffer_size=100,
        alpha=0.2,
        beta=0.2,
        e_lambda=0.1,
        gamma=0.9,
        softmax_temp=2.0,
        n_tasks=10,
        num_classes=27,
        task_num_classes=CLASS_COUNTS,
        class_offsets=OFFSETS,
        optim_mom=0.0,
        optim_wd=0.0,
        optim_nesterov=0,
        maxlr=0.01,
        minlr=0.001,
        fitting_epochs=1,
        backbone_max_patches=0,
    )


def _backbone():
    return GenericMILBackbone(input_dim=8, num_classes=27, hidden_dim=8)


class PerfectNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        features = inputs[0]
        label = int(features[0, 0].item())
        logits = torch.full((1, 27), -10.0, device=features.device) + self.anchor * 0
        logits[0, label] = 10.0
        return logits, torch.softmax(logits, 1), logits.argmax(1), None, logits.sum() * 0


class PerfectModel(torch.nn.Module):
    COMPATIBILITY = ["class-il", "task-il"]

    def __init__(self):
        super().__init__()
        self.net = PerfectNet()
        self.device = torch.device("cpu")

    def forward(self, inputs):
        return self.net(inputs)

    def prepare_inputs(self, features, coords, patch_size, training=None):
        return features, coords, patch_size


class SyntheticStream:
    def __init__(self, task_num_classes=CLASS_COUNTS):
        self.task_num_classes = list(task_num_classes)
        self.class_offsets = []
        offset = 0
        for count in self.task_num_classes:
            self.class_offsets.append(offset)
            offset += count
        self.test_loaders = []
        for offset, count in zip(self.class_offsets, self.task_num_classes):
            batches = []
            for label in range(offset, offset + count):
                batches.append((
                    torch.tensor([[float(label)]]),
                    torch.zeros((1, 2), dtype=torch.long),
                    torch.tensor(1024),
                    torch.tensor([label]),
                ))
            self.test_loaders.append(batches)

    def task_slice(self, task_id):
        return slice(
            self.class_offsets[task_id],
            self.class_offsets[task_id] + self.task_num_classes[task_id],
        )

    def seen_class_count(self, task_id):
        return self.task_slice(task_id).stop


class DynamicEvaluationTests(unittest.TestCase):
    def test_perfect_ten_task_class_and_task_il(self):
        result = evaluate(PerfectModel(), SyntheticStream())
        self.assertEqual(result.class_il_accuracy, [1.0] * 10)
        self.assertEqual(result.task_il_accuracy, [1.0] * 10)
        self.assertTrue(all(abs(auc - 1.0) < 1e-6 for auc in result.task_auc))
        self.assertAlmostEqual(result.global_seen_auc, 1.0)

    def test_reverse_order(self):
        stream = SyntheticStream(list(reversed(CLASS_COUNTS)))
        result = evaluate(PerfectModel(), stream)
        self.assertEqual(result.class_il_accuracy, [1.0] * 10)
        self.assertEqual(result.task_il_accuracy, [1.0] * 10)


class DynamicMethodSmokeTests(unittest.TestCase):
    def setUp(self):
        self.args = _args()
        self.features = torch.randn(8, 8)
        self.coords = torch.zeros((8, 2), dtype=torch.long)
        self.patch_size = torch.tensor(1024)

    def test_all_methods_first_step(self):
        methods = [AGem, ConSlide, Derpp, ErACE, EwcOn, GDumb, Lwf, Sgd]
        for method in methods:
            with self.subTest(method=method.__name__):
                model = method(_backbone(), torch.nn.functional.cross_entropy, self.args, None)
                loss = model.observe(
                    self.features, self.coords, self.patch_size,
                    torch.tensor([0]), 0, ssl=False,
                )
                self.assertTrue(torch.isfinite(torch.as_tensor(loss)).item())
        self.assertEqual(Joint(_backbone(), torch.nn.functional.cross_entropy, self.args, None).observe(), 0.0)

    def test_replay_and_distillation_later_task(self):
        conslide = ConSlide(_backbone(), torch.nn.functional.cross_entropy, self.args, None)
        for label in range(4):
            conslide.buffer.add_data(
                [self.features, self.coords, self.patch_size], torch.tensor([label])
            )
        loss = conslide.observe(
            self.features, self.coords, self.patch_size, torch.tensor([4]), 1, ssl=False
        )
        self.assertTrue(np_isfinite(loss))

        derpp = Derpp(_backbone(), torch.nn.functional.cross_entropy, self.args, None)
        derpp.observe(self.features, self.coords, self.patch_size, torch.tensor([0]), 0, ssl=False)
        self.assertTrue(np_isfinite(derpp.observe(
            self.features, self.coords, self.patch_size, torch.tensor([4]), 1, ssl=False
        )))

        lwf = Lwf(_backbone(), torch.nn.functional.cross_entropy, self.args, None)
        lwf.begin_task(None)
        lwf.observe(self.features, self.coords, self.patch_size, torch.tensor([0]), 0, ssl=False)
        lwf.end_task(None)
        lwf.begin_task(None)
        self.assertTrue(np_isfinite(lwf.observe(
            self.features, self.coords, self.patch_size, torch.tensor([4]), 1, ssl=False
        )))

    def test_all_other_methods_later_task(self):
        agem = AGem(_backbone(), torch.nn.functional.cross_entropy, self.args, None)
        agem.buffer.add_data([self.features, self.coords, self.patch_size], torch.tensor([0]))
        self.assertTrue(np_isfinite(agem.observe(
            self.features, self.coords, self.patch_size, torch.tensor([4]), 1
        )))

        erace = ErACE(_backbone(), torch.nn.functional.cross_entropy, self.args, None)
        erace.observe(self.features, self.coords, self.patch_size, torch.tensor([0]), 0)
        erace.end_task(None)
        self.assertTrue(np_isfinite(erace.observe(
            self.features, self.coords, self.patch_size, torch.tensor([4]), 1
        )))

        ewc = EwcOn(_backbone(), torch.nn.functional.cross_entropy, self.args, None)
        ewc.checkpoint = ewc.net.get_params().detach().clone()
        ewc.fish = torch.ones_like(ewc.checkpoint)
        self.assertTrue(np_isfinite(ewc.observe(
            self.features, self.coords, self.patch_size, torch.tensor([4]), 1
        )))

        gdumb = GDumb(_backbone(), torch.nn.functional.cross_entropy, self.args, None)
        self.assertTrue(np_isfinite(gdumb.observe(
            self.features, self.coords, self.patch_size, torch.tensor([4]), 1
        )))

        sgd = Sgd(_backbone(), torch.nn.functional.cross_entropy, self.args, None)
        self.assertTrue(np_isfinite(sgd.observe(
            self.features, self.coords, self.patch_size, torch.tensor([4]), 1, False
        )))

    def test_instance_buffer_handles_2_3_4_5_class_tasks(self):
        buffer = Instance_Buffer(500, torch.device("cpu"))
        starts_and_counts = [(0, 2), (2, 3), (5, 4), (9, 5)]
        for start, count in starts_and_counts:
            for label in range(start, start + count):
                buffer.add_data(
                    [self.features, self.coords, self.patch_size], torch.tensor([label])
                )
            bags = buffer.get_task_bags(range(start, start + count), size=3)
            self.assertEqual(len(bags), count)


def np_isfinite(value):
    return bool(torch.isfinite(torch.as_tensor(value)).item())


if __name__ == "__main__":
    unittest.main()
