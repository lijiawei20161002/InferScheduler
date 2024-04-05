def extract_selected_ids(log_contents):
    """
    Extract selected request IDs from log contents.
    Returns a dictionary where the key is the iteration number
    and the value is the set of selected request IDs.
    """
    selected_ids_per_iteration = {}
    iteration = None
    for line in log_contents.split('\n'):
        if line.startswith('Iteration:'):
            iteration = int(line.split(',')[0].split(':')[1].strip())
        if line.startswith('Selected Requests:'):
            ids_text = line
            if len(ids_text.split('{'))>1:
                print(ids_text.split('{')[1:])
                ids = {req.split("'")[3] for req in ids_text.split('{')[1:]}  # Extracting IDs
            else:
                ids = {}
            selected_ids_per_iteration[iteration] = ids
    return selected_ids_per_iteration

def compare_logs(file_path1, file_path2):
    # Read log file contents
    with open(file_path1, 'r') as file1, open(file_path2, 'r') as file2:
        log1_contents = file1.read()
        log2_contents = file2.read()

    # Extract selected request IDs
    log1_selected_ids = extract_selected_ids(log1_contents)
    log2_selected_ids = extract_selected_ids(log2_contents)

    # Compare and find first iteration with differences
    for iteration in sorted(set(log1_selected_ids.keys()) | set(log2_selected_ids.keys())):
        ids1 = log1_selected_ids.get(iteration, set())
        ids2 = log2_selected_ids.get(iteration, set())

        if ids1 != ids2:
            print(f"Difference found at iteration {iteration}.")
            print(f"File 1 IDs: {ids1}")
            print(f"File 2 IDs: {ids2}")
            break
    else:
        print("No differences found in selected request IDs.")

file_path1 = 'online solver.log'
file_path2 = 'offline solver.log'
compare_logs(file_path1, file_path2)
