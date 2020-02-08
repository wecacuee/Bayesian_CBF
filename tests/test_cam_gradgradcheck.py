from pathlib import Path
from functools import partial

import torch
import pytest

from bayes_cbf.control_affine_model import ControlAffineRegressor

__directory__ = Path(__file__).parent or Path(".")

@pytest.mark.skip('regenerate pth file')
def test_cam_gradgradcheck(
        saved_model_path=__directory__ / 'data' / 'cam-gradgradcheck-fail.pth'):
    D = torch.load(str(saved_model_path))
    model_dict = D['model']
    Xtrain = D['xtrain']
    Xtest = D['xtest']
    car = ControlAffineRegressor(Xtrain.shape[-1], 1)
    car.model.load_state_dict(model_dict)
    car.to(dtype=torch.float64)
    torch.autograd.gradgradcheck(
                    partial(lambda s, Xt, X: s.model.covar_module.data_covar_module(
                        X, X).evaluate(), car, Xtrain),
                    Xtest.double(),
                    raise_exception=True)

if __name__ == '__main__':
    test_cam_gradgradcheck()
