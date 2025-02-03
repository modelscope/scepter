# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING
from scepter.modules.utils.import_utils import LazyImportModule


if TYPE_CHECKING:
    from scepter.modules.solver import hooks
    from scepter.modules.solver.base_solver import BaseSolver
    from scepter.modules.solver.diffusion_solver import LatentDiffusionSolver
    from scepter.modules.solver.train_val_solver import TrainValSolver
    from scepter.modules.solver.ace_solver import ACESolver
    from scepter.modules.solver.ace_plus_solver import ACEPlusSolver
    from scepter.modules.solver.diffusion_video_solver import LatentDiffusionVideoSolver
else:
    _import_structure = {
        'solver': ['hooks'],
        'base_solver': ['BaseSolver'],
        'diffusion_solver': ['LatentDiffusionSolver'],
        'train_val_solver': ['TrainValSolver'],
        'ace_solver': ['ACESolver'],
        'ace_plus_solver': ['ACEPlusSolver'],
        'diffusion_video_solver': ['LatentDiffusionVideoSolver']
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
