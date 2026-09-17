import torch
from torch import nn

from src.data.processed_bundle import make_synthetic_bundle
from src.models.mingle import MingleModel
from src.models.pma import PMA
from src.training.train_concept_ablation import zero_concept_semantics
from src.training.validate_architecture import masked_bce_with_logits, run_forward


def test_pma_is_custom() -> None:
    pma = PMA(48, 4)
    assert not isinstance(pma, nn.MultiheadAttention)


def test_forward_shapes_and_masks() -> None:
    bundle = make_synthetic_bundle()
    model, outputs, loss = run_forward(bundle)

    assert isinstance(model, MingleModel)
    assert outputs["logits"].shape == (bundle.num_real, 25)
    assert outputs["jk"].shape == (bundle.num_real, 96)
    assert outputs["h_e"].shape[0] == bundle.num_real + bundle.num_self_loops
    assert outputs["e_e"].shape[0] == bundle.num_real + bundle.num_self_loops
    assert outputs["x_v"].shape[0] == len(bundle.node_ids)
    assert bundle.num_self_loops == len(bundle.node_ids)
    assert int(bundle.pair_mask.sum()) == 3
    assert bool(bundle.pair_mask[2]) is False
    assert bool(bundle.pair_mask[4]) is False
    assert bundle.labels[2].sum() == 0
    assert bundle.labels[4].sum() == 0
    assert loss.ndim == 0
    assert not loss.isnan()


def test_concept_ablation_zeros_cv_and_keeps_deepwalk() -> None:
    bundle = make_synthetic_bundle()
    walk = bundle.node_states[:, :64].clone()
    zero_concept_semantics(bundle)
    assert torch.count_nonzero(bundle.concept_semantics) == 0
    assert torch.count_nonzero(bundle.node_states[:, -768:]) == 0
    assert torch.equal(bundle.node_states[:, :64], walk)


def test_targets_come_from_next_visit_only() -> None:
    bundle = make_synthetic_bundle()
    # e0 -> e1 should be a valid pair; e2 is terminal for p0.
    assert bool(bundle.pair_mask[0])
    assert bool(bundle.pair_mask[1])
    assert not bool(bundle.pair_mask[2])
