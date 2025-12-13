# load_chronos_compat.py
import inspect
from chronos import ChronosPipeline
import chronos
# Grab ChronosConfig class
try:
    ChronosConfig = chronos.chronos.ChronosConfig
except Exception:
    # fallback path if package structure differs
    ChronosConfig = getattr(chronos, "ChronosConfig", None)

if ChronosConfig is None:
    raise SystemExit("Could not locate ChronosConfig in chronos package. Aborting.")

# Save original __init__
_orig_init = ChronosConfig.__init__

# Get accepted kwargs (parameter names) of original __init__
_sig = inspect.signature(_orig_init)
accepted_params = set(list(_sig.parameters.keys())[1:])  # drop 'self'

def _patched_init(self, *args, **kwargs):
    # Filter kwargs: keep only the parameters the original __init__ accepts
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}
    # Optionally log removed keys for debugging
    removed = set(kwargs.keys()) - set(filtered_kwargs.keys())
    if removed:
        # print removed keys exactly once to avoid spam
        print(f"[chronos-compat] removed unknown config keys: {sorted(list(removed))}")
    return _orig_init(self, *args, **filtered_kwargs)

# Apply patch
ChronosConfig.__init__ = _patched_init

# Now load the model (local or remote). Prefer local folder if you already downloaded.
# Example: local path './chronos_bolt' (replace with your actual path)
model_path = "./chronos_bolt"  # or "amazon/chronos-bolt-base" if internet works
print("Attempting to load Chronos from:", model_path)
model = ChronosPipeline.from_pretrained(model_path)

print("Chronos model loaded successfully with compatibility shim.")
# You can now use `model` for predict calls
# Example (pseudo):
# preds = model.predict( ... )
