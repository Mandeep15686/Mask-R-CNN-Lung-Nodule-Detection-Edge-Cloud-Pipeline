"""Lung nodule Mask R-CNN training package."""
from .losses import CompositeMaskLoss, BCELoss, SoftDiceLoss, BoundaryLoss, FocalLoss, HausdorffDTLoss
from .model import LungNoduleMaskRCNN, EdgeFeatureExtractor, build_model
from .trainer import Trainer, MGDATrainer, build_trainer
from .mgda import MGDASolver, MGDASolution, MGDAResult, solve_mgda
from .metrics import evaluate_batch, compute_froc, MetricTracker
from .experiment import ExperimentRunner, BASE_CONFIG
from .phase3_bayesian import (
    run_phase3_bayesian,
    normalise_weights,
    compute_objective,
    Phase3TrialTrainer,
)
