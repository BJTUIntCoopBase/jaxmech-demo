"""Solid elastic analysis helpers for the demo subset."""

__all__ = ["export_solid_elastic_result_mat", "solve_linear_step"]


def export_solid_elastic_result_mat(*args, **kwargs):
    from jaxmech.modules.inc_analysis.elastic.export_mat import export_solid_elastic_result_mat as impl

    return impl(*args, **kwargs)


def solve_linear_step(*args, **kwargs):
    from jaxmech.modules.inc_analysis.elastic.linear import solve_linear_step as impl

    return impl(*args, **kwargs)
