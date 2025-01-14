import json
from pathlib import Path
import streamlit as st

DATA_PATH = Path("data/codearena_instances.jsonl")

# Load data from the JSONL file
def load_data(file_path):
    return [json.loads(line) for line in file_path.read_text().splitlines()]

# Save updated data to the JSONL file
def save_changes(data, file_path):
    with open(file_path, 'w') as file:
        file.writelines(json.dumps(entry) + '\n' for entry in data)

# Main application logic
def main():
    st.title("CodeArena Browser")

    # Load data
    data = load_data(DATA_PATH)

    # Extract instance IDs for the selector
    instance_ids = [entry['instance_id'] for entry in data]
    selected_instance_id = st.selectbox("Select an instance_id", instance_ids)

    # Get the selected instance and make it editable
    selected_instance = next((entry for entry in data if entry['instance_id'] == selected_instance_id), None)
    if not selected_instance:
        st.error("No data found for the selected instance_id.")
        return

    # Initialize editable instance in session state
    if "editable_instance" not in st.session_state:
        st.session_state.editable_instance = selected_instance.copy()

    # Save changes button
    if st.button("Save Changes"):
        # Update the original data with the edited instance
        for i, entry in enumerate(data):
            if entry['instance_id'] == selected_instance_id:
                data[i] = st.session_state.editable_instance  # Replace the instance with the edited copy
                break

        # Write the updated data back to disk
        save_changes(data, DATA_PATH)
        st.success("Changes saved successfully!")

    # Display and edit data for the selected instance
    st.markdown(f"Instance: `{selected_instance_id}`")

    for key, value in st.session_state.editable_instance.items():
        if isinstance(value, str):
            # Editable text fields
            st.session_state.editable_instance[key] = st.text_area(
                key, value, height=200 if len(value) > 200 else None
            )
        elif isinstance(value, (list, dict)):
            # Show JSON as editable JSON strings
            json_value = json.dumps(value, indent=2)
            json_text = st.text_area(key, json_value, height=200)
            try:
                st.session_state.editable_instance[key] = json.loads(json_text)  # Parse back into JSON
            except json.JSONDecodeError:
                st.error(f"Invalid JSON for {key}. Please fix it.")

    # Add a key-value pair to the instance
    with st.expander("Add Key-Value Pair"):
        new_key = st.text_input("Key")
        new_value = st.text_input("Value (will be treated as string)")
        if st.button("Add Key-Value Pair"):
            if new_key and new_key not in st.session_state.editable_instance:
                st.session_state.editable_instance[new_key] = new_value
                st.success(f"Key '{new_key}' added successfully!")
            elif new_key in st.session_state.editable_instance:
                st.error(f"Key '{new_key}' already exists!")
            else:
                st.error("Key cannot be empty!")

if __name__ == "__main__":
    main()
