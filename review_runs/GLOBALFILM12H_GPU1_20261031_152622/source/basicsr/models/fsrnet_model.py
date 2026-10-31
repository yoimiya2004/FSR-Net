"""Paper-facing trainer name with legacy checkpoint compatibility."""
from basicsr.utils.registry import MODEL_REGISTRY
from .femasr_model import FeMaSRModel

@MODEL_REGISTRY.register()
class FSRNetModel(FeMaSRModel):
    pass
