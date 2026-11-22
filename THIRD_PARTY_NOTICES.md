# Third-party notices

orbq-mlip is licensed under the Apache License 2.0 (see `LICENSE`). It includes
code adapted from the following projects, which are distributed under the MIT
license reproduced below.

| Project | Files in this repository | Copyright |
|---|---|---|
| [fairchem](https://github.com/facebookresearch/fairchem) (formerly Open Catalyst Project) | `orbq/model/scn/` (eSCN), `orbq/model/utils.py` (parts), `orbq/common/typing.py` | Copyright (c) Meta, Inc. and its affiliates. |
| [scdp](https://github.com/kyonofx/scdp) | `orbq/model/gtos.py`, `orbq/model/basis_set.py` | Copyright (c) 2024 Xiang |
| [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) | `orbq/common/pyg/` (trimmed subset) | Copyright (c) 2023 PyG Team |

The bundled test sample `tests/data/omol25_sample/` is derived from the
[OMol25](https://huggingface.co/facebook/OMol25) dataset (CC BY 4.0); see its README.

## MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
