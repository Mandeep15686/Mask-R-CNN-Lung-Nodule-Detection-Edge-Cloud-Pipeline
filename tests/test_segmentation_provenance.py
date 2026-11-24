import ast
from pathlib import Path


def test_segmentation_metric_provenance_is_explicit_and_scoped():
    source_path = Path(__file__).parents[1] / "training" / "trainer.py"
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    provenance_node = next(
        node for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "SEGMENTATION_PROVENANCE"
            for target in node.targets
        )
    )
    provenance = ast.literal_eval(provenance_node.value)
    assert provenance["segmentation_target"] == (
        "synthetic_spherical_approximation_from_center_and_diameter"
    )
    assert provenance["segmentation_metric_scope"] == "sampled_patch_validation"
    assert "pseudo-masks" in provenance["segmentation_metric_interpretation"]
    assert "not expert-contour accuracy" in provenance["segmentation_metric_interpretation"]
    assert 'self.config.update(SEGMENTATION_PROVENANCE)' in source
    assert "**SEGMENTATION_PROVENANCE" in source
    assert '"config": self.config' in source  # best checkpoint metadata
