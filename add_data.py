from pathlib import Path
import json
import importlib
import copy

from unidiff import PatchSet
import streamlit as st
from streamlit_ace import st_ace

import swebench


def get_modified_files(patch_str: str) -> list[str]:
    patchset = PatchSet(patch_str)
    return [f.path for f in patchset.modified_files]


DATA_PATH = Path("data/codearena_instances.json")
REPO_DATA_PATH = Path("data/codearena_repo_data.py")


def main():
    st.title("CodeArena Data Loader")
    
    data = json.loads(DATA_PATH.read_text())

    # Editable field for REPO_DATA
    st.subheader("REPO_DATA")
    repo_data_code = REPO_DATA_PATH.read_text()
    
    # Display REPO_DATA code in an ACE code editor with syntax highlighting
    new_repo_data_code = st_ace(
        value=repo_data_code,
        language='python',
        theme='xcode',
        height=500,
        key='repo_data_editor',
        auto_update=True,
    )

    # Save button to save changes to REPO_DATA
    if st.button("Save REPO_DATA Changes"):
        with open(REPO_DATA_PATH, 'w') as f:
            f.write(new_repo_data_code)
        st.success("REPO_DATA changes saved successfully!")

    REPO_DATA = eval(REPO_DATA_PATH.read_text())
    
    if REPO_DATA is None:
        st.error(f"REPO_DATA could not be loaded from {REPO_DATA_PATH}")
    else:

        file_path = st.text_input("Path to instances data:")

        if st.button("Check instances"):
            if file_path:
                input_data = [json.loads(l) for l in Path(file_path).read_text().splitlines()]
                
                repo_data_unspecified, checked_instances = set(), []
                
                for instance in input_data:
                    instance_repo = instance['repo']

                    if instance_repo not in REPO_DATA:
                        repo_data_unspecified.add(instance_repo)
                        continue

                    swebench.versioning.constants.MAP_REPO_TO_VERSION_PATHS[instance_repo] = REPO_DATA[instance_repo]["MAP_REPO_TO_VERSION_PATHS"]
                    swebench.versioning.constants.MAP_REPO_TO_VERSION_PATTERNS[instance_repo] = REPO_DATA[instance_repo]["MAP_REPO_TO_VERSION_PATTERNS"]

                    importlib.reload(swebench)

                    instance_version = swebench.versioning.get_versions.get_version(
                        instance=instance,
                        is_build=False,
                        path_repo=None,
                    )

                    specs_available = instance_version in REPO_DATA[instance_repo]["MAP_REPO_VERSION_TO_SPECS"]
                    checked_instances.append({
                        "instance": instance,
                        "version": instance_version,
                        "specs_available": specs_available,
                    })

                if len(repo_data_unspecified) == 0:
                    st.success(f"REPO_DATA specified for all repositories")
                else:
                    st.error(f"REPO_DATA not specified for the following repositories:\n{'\n'.join(repo_data_unspecified)}")


                st.dataframe([
                    {
                        "instance_id": d['instance']['instance_id'],
                        "specs_available": d['specs_available'],
                        "link": f"https://github.com/{d['instance']['repo']}/tree/{d['instance']['base_commit']}",
                        "version": d['version'],
                        "base_commit": d['instance']['base_commit'],
                    }
                    for d in checked_instances
                ])
                
            else:
                st.warning("Please enter a valid file path.")


        if st.button("Check and save data"):
            if file_path:
                input_data = [json.loads(l) for l in Path(file_path).read_text().splitlines()]
                
                repo_data_unspecified, checked_instances = set(), []
                
                for instance in input_data:
                    instance_repo = instance['repo']

                    if instance_repo not in REPO_DATA:
                        repo_data_unspecified.add(instance_repo)
                        continue

                    swebench.versioning.constants.MAP_REPO_TO_VERSION_PATHS[instance_repo] = REPO_DATA[instance_repo]["MAP_REPO_TO_VERSION_PATHS"]
                    swebench.versioning.constants.MAP_REPO_TO_VERSION_PATTERNS[instance_repo] = REPO_DATA[instance_repo]["MAP_REPO_TO_VERSION_PATTERNS"]

                    importlib.reload(swebench)

                    instance_version = swebench.versioning.get_versions.get_version(
                        instance=instance,
                        is_build=False,
                        path_repo=None,
                    )

                    specs_available = instance_version in REPO_DATA[instance_repo]["MAP_REPO_VERSION_TO_SPECS"]
                    checked_instances.append({
                        "instance": instance,
                        "version": instance_version,
                        "specs_available": specs_available,
                    })


                processed_instances = []
                for ci in checked_instances:
                    if ci["specs_available"]:
                        instance = copy.deepcopy(ci['instance'])
                        instance["version"] = ci['version']
                        instance["PASS_TO_PASS"] = []                
                        instance["FAIL_TO_PASS"] = get_modified_files(instance['test_patch'])
                        processed_instances.append(instance)

                data.extend(processed_instances)

                DATA_PATH.write_text(json.dumps(data, indent=4))
                st.success(f"Added {len(processed_instances)} instances successfully to {DATA_PATH}")

            else:
                st.warning("Please enter a valid file path.")


def patch_constants(module_to_patch, repo_data):
    for repo, data in repo_data.items():
        module_to_patch.versioning.constants.MAP_REPO_TO_VERSION_PATHS[repo] = data["MAP_REPO_TO_VERSION_PATHS"]
        module_to_patch.versioning.constants.MAP_REPO_TO_VERSION_PATTERNS[repo] = data["MAP_REPO_TO_VERSION_PATTERNS"]
        module_to_patch.harness.constants.MAP_REPO_VERSION_TO_SPECS[repo] = data["MAP_REPO_VERSION_TO_SPECS"]


if __name__ == '__main__':
    main()
    