"""Compatibility elastic subpackage for incremental analysis."""

__all__ = [
    "export_shell_elastic_result_mat",
    "export_solid_elastic_result_mat",
    "solve_linear_step",
]


def export_shell_elastic_result_mat(*args, **kwargs):
    """Lazy compatibility wrapper for the shell elastic MAT exporter."""
    from jaxmech.modules.inc_analysis.elastic.export_mat import export_shell_elastic_result_mat as impl

    return impl(*args, **kwargs)


def export_solid_elastic_result_mat(*args, **kwargs):
    """Lazy compatibility wrapper for the solid elastic MAT exporter."""
    from jaxmech.modules.inc_analysis.elastic.export_mat import export_solid_elastic_result_mat as impl

    return impl(*args, **kwargs)


def solve_linear_step(*args, **kwargs):
    """Lazy compatibility wrapper for the solid linear step solver."""
    from jaxmech.modules.inc_analysis.elastic.linear import solve_linear_step as impl

    return impl(*args, **kwargs)