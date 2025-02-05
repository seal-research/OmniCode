from pathlib import Path
import json
import importlib

from unidiff import PatchSet
import streamlit as st
from streamlit_ace import st_ace

import swebench


def get_modified_files(patch_str: str) -> list[str]:
    patchset = PatchSet(patch_str)
    return [f.path for f in patchset.modified_files]


DATA_PATH = Path("data/codearena_instances.jsonl")
REPO_DATA_PATH = Path("data/codearena_repo_data.py")


def main():
    st.title("CodeArena Data Loader")

    data = [json.loads(l) for l in DATA_PATH.read_text().splitlines()]

    # Editable field for REPO_DATA
    st.subheader("REPO_DATA")
    repo_data_code = REPO_DATA_PATH.read_text()

    # Display REPO_DATA code in an ACE code editor with syntax highlighting
    new_repo_data_code = st_ace(
        value=repo_data_code,
        language="python",
        theme="xcode",
        height=500,
        key="repo_data_editor",
        auto_update=True,
    )

    # Save button to save changes to REPO_DATA
    if st.button("Save REPO_DATA Changes"):
        with open(REPO_DATA_PATH, "w") as f:
            f.write(new_repo_data_code)
        st.success("REPO_DATA changes saved successfully!")

    REPO_DATA = eval(REPO_DATA_PATH.read_text())

    if REPO_DATA is None:
        st.error(f"REPO_DATA could not be loaded from {REPO_DATA_PATH}")
    else:

        file_path = st.text_input("Path to instances data:")

        if st.button("Process instances"):
            if file_path:
                input_data = [
                    json.loads(l) for l in Path(file_path).read_text().splitlines()
                ]

                # process each instance, add to data and save to disk
                processed_instances = []
                clean = True

                for instance in input_data:

                    instance_repo = instance["repo"]

                    if instance_repo not in REPO_DATA:
                        st.error(f"REPO_DATA not specified for {instance_repo}")
                        clean = False
                        break

                    swebench.versioning.constants.MAP_REPO_TO_VERSION_PATHS[
                        instance_repo
                    ] = REPO_DATA[instance_repo]["MAP_REPO_TO_VERSION_PATHS"]
                    swebench.versioning.constants.MAP_REPO_TO_VERSION_PATTERNS[
                        instance_repo
                    ] = REPO_DATA[instance_repo]["MAP_REPO_TO_VERSION_PATTERNS"]

                    importlib.reload(swebench)

                    instance_version = swebench.versioning.get_versions.get_version(
                        instance=instance,
                        is_build=False,
                        path_repo=None,
                    )
                    # print(f"version= {instance_version}")
                    # print(
                    #     f"instance_id: {instance['instance_id']}, commit: {instance['base_commit']}"
                    # )
                    if (
                        instance_version
                        not in REPO_DATA[instance_repo]["MAP_REPO_VERSION_TO_SPECS"]
                    ):
                        st.error(
                            f"SPECS not found for version {instance_version} of repo {instance_repo} in REPO_DATA"
                        )
                        clean = False
                        break

                    instance["version"] = instance_version
                    instance["PASS_TO_PASS"] = []
                    instance["FAIL_TO_PASS"] = get_modified_files(
                        instance["test_patch"]
                    )

                    # Add to the processed instances list
                    processed_instances.append(instance)

                if clean:

                    st.success("Instances processed successfully")
                    data.extend(processed_instances)

                    with open(DATA_PATH, "w") as f:
                        for instance in data:
                            f.write(json.dumps(instance) + "\n")

                    st.success(
                        f"Added {len(processed_instances)} instances successfully"
                    )

            else:
                st.warning("Please enter a valid file path.")

        # if st.button("Save to CodeArena"):

        #     print("Adding data")
        #     with open(DATA_PATH, 'w') as f:
        #         for instance in data:
        #             f.write(json.dumps(instance) + '\n')

        #     print(f"Added {len(processed_instances)} instances successfully")
        #     st.stop()


def patch_constants(module_to_patch, repo_data):
    for repo, data in repo_data.items():
        module_to_patch.versioning.constants.MAP_REPO_TO_VERSION_PATHS[repo] = data[
            "MAP_REPO_TO_VERSION_PATHS"
        ]
        module_to_patch.versioning.constants.MAP_REPO_TO_VERSION_PATTERNS[repo] = data[
            "MAP_REPO_TO_VERSION_PATTERNS"
        ]
        module_to_patch.harness.constants.MAP_REPO_VERSION_TO_SPECS[repo] = data[
            "MAP_REPO_VERSION_TO_SPECS"
        ]


if __name__ == "__main__":
    main()
