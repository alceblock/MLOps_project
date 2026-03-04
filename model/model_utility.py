import os
import re

## --- choose best model from version (use config file to keep track of best)
# TODO
## --- END - choose best model from version (use config file to keep track of best)

# ---- save new model version
BASE_PATH = "./model_versions"

def get_version():
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)

    existing = [
        d for d in os.listdir(BASE_PATH)
        if re.match(r"my_finetuned_model\d+", d)
    ]

    if not existing:
        return 0

    versions = [
        int(re.findall(r"\d+", d)[0])
        for d in existing
    ]

    return max(versions)

def build_next_version(version):
    return version + 1


def save_new_version():
    version = build_next_version(get_version())
    save_path = os.path.join(BASE_PATH, f"my_finetuned_model{version}")

    print(f"Saved version {version}")
    return save_path
# trainer.save_model(save_path)
# tokenizer.save_pretrained(save_path)

# ---- END - save new model version