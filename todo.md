# TODO — Wheel → manylinux → PyPI for `opencl_ocl`

## Goal

1. Build a real **`.whl`**
2. Produce **manylinux**-compatible wheels (broader Linux installs)
3. **Upload to PyPI** so:

```bash
pip install opencl-ocl
```

works without grabbing a raw `.so` from GitHub.

---

## Phase 0 — already true

- [x] At least one external download / interest signal
- [x] Extension builds locally (`.so`)
- [x] ATen fallback + bounded OpenCL cache

---

## Phase 1 — local wheel (block PyPI until this is green)

- [x] Package layout + `pyproject.toml` (name, version, MIT, `requires-python`)
- [x] CMake/pybind build integrated with setuptools (adapt `setup_ocl.py`)
- [x] Import name stays `opencl_ocl` for `text_trainer.py`
- [x] `python -m build --wheel` on your machine
- [x] Clean venv: `pip install dist/*.whl` → `import opencl_ocl; opencl_ocl.is_available()`
- [x] Trainer runs with wheel install only (no manual `.so` copy)
- [x] Built sdist and uploaded to TestPyPI and production PyPI (`opencl-ocl 0.1.1` live)

**PyPI name:** `opencl-ocl`  
**Import name:** `opencl_ocl`

---

## Phase 2 — manylinux (Linux portability)

manylinux = wheel built inside a **CentOS/Alma-based container** with old glibc so it runs on most modern Linux distros.

- [x] Use **cibuildwheel** or official manylinux image
  - Config added to `pyproject.toml` under `[tool.cibuildwheel]`
- [x] Config in `pyproject.toml`
  - Build matrix: `cp310-* cp311-* cp312-*`, archs `x86_64`
- [x] Torch dependency strategy
  - `install_requires=["torch>=2.0"]` in `pyproject.toml`
- [ ] Repair wheel with `auditwheel repair` (cibuildwheel does this automatically)
- [ ] Smoke test manylinux wheel on a **different** Linux box/VM (not the build host)

### OpenCL + manylinux reality check

| Piece | In the wheel? |
|--------|----------------|
| Your `.so` ops | Yes |
| libOpenCL / ICD loader | Sometimes (careful with auditwheel) |
| Vendor GPU driver | **Never** — user installs |
| Works headless CPU-only | Yes via ATen if you keep fallback |

Document: **“Needs OpenCL ICD for GPU path; CPU/ATen always works.”**

---

## Phase 3 — PyPI upload

- [x] Create account on **https://pypi.org** (+ enable 2FA)
- [x] Test upload to **TestPyPI** first
  - `opencl-ocl 0.1.1` uploaded successfully to TestPyPI
- [x] Project metadata on PyPI
  - Author: `tttesre <ttes2@proton.me>`
  - Long description = README
  - License: MIT
- [x] Production upload
  - `opencl-ocl 0.1.1` live at https://pypi.org/project/opencl-ocl/0.1.1/
- [x] Version bump policy: `0.1.0` → `0.1.1` for fixes; no overwrite of published versions

---

## Phase 4 — CI (so you're not the only builder)

- [x] GitHub Actions workflow created (`.github/workflows/wheels.yml`)
  - On tag `v*`: run cibuildwheel → upload artifacts
  - Publishes to PyPI with `PYPI_API_TOKEN` secret
- [ ] Matrix: Python 3.10–3.12, `manylinux_2_28_x86_64` (or 2_17 if you need older)
- [ ] Workflow validated by pushing a test tag and confirming build succeeds

---

## Acceptance

- [ ] `pip install opencl-ocl` from PyPI on a clean Linux x86_64 + Python 3.12
- [ ] `import opencl_ocl` works
- [ ] Without OpenCL device: no crash, `is_available() == False`, ATen path usable
- [ ] With OpenCL ICD: kernels run or soft-fallback
- [ ] manylinux wheel installs on Ubuntu/Fedora/Arch-class systems without rebuild

---

## Order of work (don’t skip)

1. Local wheel + clean venv install  
2. cibuildwheel manylinux on GitHub Actions  
3. TestPyPI  
4. PyPI  
5. README: `pip install torch && pip install opencl-ocl`

---

## Notes

- **One download is enough** to justify packaging; PyPI is how the next ten don’t need a `.so` handoff  
- manylinux fixes “built on Arch, won’t load on Ubuntu” glibc issues — not Windows/macOS  
- Windows/macOS wheels are separate later (different OpenCL stacks)  
- Never commit PyPI tokens; use API tokens + GH secrets only  

---

## CPU Memory GC (Python-side hygiene)

- [x] `gc.collect()` after each PPO update block in `_train_single`
- [x] `gc.collect()` after each PPO update block in `_train_vectorized`
- [x] `del flat_trajectories, last_values` in `_train_vectorized` after update
- [x] Existing `del` + `gc.collect()` in `PPOAgent.update` retained  