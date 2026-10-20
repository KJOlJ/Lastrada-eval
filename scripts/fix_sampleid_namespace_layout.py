"""Handle pip's editable VCS checkout layout for Sony SampleID."""

from pathlib import Path


path = Path(__file__).resolve().parents[1] / "src/lastrada/recognition/sampleid_adapter.py"
text = path.read_text(encoding="utf-8")

old = '''def _prepare_sampleid_model_imports():
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
'''

new = '''def _prepare_sampleid_model_imports():
    """Expose Sony's private checkpoint modules as top-level ``src``.

    Pip may clone an editable VCS dependency into this project's ``src/sampleid``
    directory. Because Lastrada itself uses a ``src`` layout, ``import sampleid``
    can then resolve to a namespace package whose ``__file__`` is ``None``. Find
    the real nested Sony package in either layout, put its directory first on
    ``sys.path``, and clear any unrelated top-level ``src`` namespace before
    Hydra instantiates checkpoint classes such as
    ``src.networks.resnet_ibn.ResnetIBN``.
    """
    import sampleid as imported_sampleid

    package_root = None
    SampleID = getattr(imported_sampleid, "SampleID", None)
    module_file = getattr(imported_sampleid, "__file__", None)
    if module_file:
        package_root = Path(module_file).resolve().parent
    else:
        for namespace_entry in getattr(imported_sampleid, "__path__", []):
            namespace_root = Path(namespace_entry).resolve()
            nested_package = namespace_root / "sampleid"
            if (nested_package / "__init__.py").is_file():
                actual_package = importlib.import_module("sampleid.sampleid")
                SampleID = getattr(actual_package, "SampleID", None)
                package_root = nested_package
                break

    if package_root is None or SampleID is None:
        raise SampleIDUnavailable(
            "Sony SampleID's installed source package could not be located. "
            "Reinstall the pinned upstream repository in editable/source form."
        )

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

    return SampleID
'''

if old not in text:
    raise RuntimeError("SampleID import helper anchor was not found")
text = text.replace(old, new, 1)
compile(text, str(path), "exec")
path.write_text(text, encoding="utf-8")
print("Sony editable namespace layout is now supported")
