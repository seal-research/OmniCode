import json
from pathlib import Path
import streamlit as st

DATA_PATH = Path("data/codearena_instances.json")

def load_data(file_path):
    return json.loads(file_path.read_text())

def save_changes(data, file_path):
    file_path.write_text(json.dumps(data, indent=4))

def main():
    st.title("CodeArena Browser")

    data = load_data(DATA_PATH)

    instance_ids = [entry['instance_id'] for entry in data]
    selected_instance_id = st.selectbox("Select an instance_id", instance_ids)

    selected_instance = next((entry for entry in data if entry['instance_id'] == selected_instance_id), None)
    if not selected_instance:
        st.error("No data found for the selected instance_id.")
        return

    st.session_state.editable_instance = selected_instance.copy()

    if st.button("Save Changes"):
        for i, entry in enumerate(data):
            if entry['instance_id'] == selected_instance_id:
                data[i] = st.session_state.editable_instance  # Replace the instance with the edited copy
                break

        save_changes(data, DATA_PATH)
        st.success("Changes saved successfully!")

    st.markdown(f"Instance: `{selected_instance_id}`")
    st.markdown(f"base_commit: `{selected_instance['base_commit']}`")
    st.text_area(f"problem_statement", selected_instance['problem_statement'], height=500)
    st.text_area(f"test_patch", selected_instance['test_patch'], height=500)
    st.text_area(f"patch", selected_instance['patch'], height=500)

    st.markdown("### Bad Patches")
    if 'bad_patches ' in st.session_state.editable_instance:
        for idx, bad_patch in enumerate(st.session_state.editable_instance['bad_patches']):
            st.text_area(f"bad_patch {idx}", bad_patch, height=500)

    # Add a new review
    with st.expander("Add a Bad Patch"):
        new_bad_patch = st.text_area("Bad Patch", height=500)
        if st.button("Add Bad Patch"):
            if 'bad_patches' not in st.session_state.editable_instance:
                st.session_state.editable_instance['bad_patches'] = []
            st.session_state.editable_instance['bad_patches'].append(new_bad_patch)

if __name__ == "__main__":
    main()
