from envs.colorblind_env.client import CBAEnv
from envs.colorblind_env.models import CBAAction, FixType

with CBAEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
    print("Reset done:", result.observation.hex_code_per_category)
    
    result = env.step(CBAAction(
        target="Class A",
        fix_type=FixType.RECOLOR,
        change_hex="#0077BB",
        change_shape=None
    ))
    print("Step done — reward:", result.reward)
    print("Done:", result.done)