"""Make Sony's checkpoint `src.networks.*` imports resolve reliably.

The upstream repository installs the public ``sampleid`` package, while the
checkpoint's Hydra config imports model classes from a sibling ``src`` package.
This patch puts the installed SampleID package directory first on ``sys.path``
and clears any unrelated namespace package named ``src`` before loading the
checkpoint.
"""

from pathlib import Path


path = Path(__file__).resolve().parents[1] / "src/lastrada/recognition/sampleid_adapter.py"
text = path.read_text(encoding="utf-8")

imports_old = '''import hashlib
import importlib.util
from pathlib import Path
'''
imports_new = '''import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
'''

class_anchor = '''

class SampleIDAdapter:
'''
helper = '''

def _prepare_sampleid_model_imports():
    """Expose Sony's private checkpoint modules as top-level ``src``.

    The checkpoint configuration names classes such as
    ``src.networks.resnet_ibn.ResnetIBN``. The upstream editable install places
    that package beneath the installed ``sampleid`` directory, while this
    project also has a repository-level directory named ``src``. Put Sony's
    package directory first and evict a previously resolved unrelated ``src``
    namespace before Hydra instantiates the model.
    """
    import sampleid

    package_root = Path(sampleid.__file__).resolve().parent
    sony_src = package_root / "src"
    if not sony_src.is_dir():
        raise SampleIDUnavailable(
            "Sony SampleID source modules are missing. Install the pinned "
            "upstream repository in editable/source form."
        )

    root_text = str(package_root)
    while root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)

    loaded_src = sys.modules.get("src")
    loaded_paths = {
        Path(value).resolve()
        for value in getattr(loaded_src, "__path__", [])
    }
    if loaded_src is not None and sony_src.resolve() not in loaded_paths:
        for module_name in list(sys.modules):
            if module_name == "src" or module_name.startswith("src."):
                sys.modules.pop(module_name, None)

    importlib.invalidate_caches()
    try:
        importlib.import_module("src.networks.resnet_ibn")
    except Exception as exc:
        raise SampleIDUnavailable(
            "Sony checkpoint model modules could not be imported from the "
            "installed SampleID source."
        ) from exc

    return sampleid.SampleID


class SampleIDAdapter:
'''

load_old = '''    def _load(self):
        if self._model is None:
            import torch
            from sampleid import SampleID

            torch.set_num_threads(2)
            try:
                self._model = SampleID.load_checkpoint(
                    str(self.checkpoint), device=torch.device("cpu")
                )
            except Exception as exc:
                raise SampleIDUnavailable(
                    f"SampleID checkpoint could not load ({type(exc).__name__})."
                ) from exc
        return self._model
'''

load_new = '''    def _load(self):
        if self._model is None:
            import torch

            SampleID = _prepare_sampleid_model_imports()
            torch.set_num_threads(2)
            try:
                self._model = SampleID.load_checkpoint(
                    str(self.checkpoint), device=torch.device("cpu")
                )
            except SampleIDUnavailable:
                raise
            except Exception as exc:
                raise SampleIDUnavailable(
                    f"SampleID checkpoint could not load ({type(exc).__name__})."
                ) from exc
        return self._model
'''

for label, old in (
    ("imports", imports_old),
    ("adapter class", class_anchor),
    ("model loader", load_old),
):
    if old not in text:
        raise RuntimeError(f"Could not patch SampleID {label}: anchor not found")

text = text.replace(imports_old, imports_new, 1)
text = text.replace(class_anchor, helper, 1)
text = text.replace(load_old, load_new, 1)
compile(text, str(path), "exec")
path.write_text(text, encoding="utf-8")
print("Sony checkpoint import path repaired")
