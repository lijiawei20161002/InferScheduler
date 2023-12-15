import matplotlib.pyplot as plt
import re

def parse_request_data(filename, request_ids, request_num):
    with open(filename, 'r') as file:
        log_data = file.read()

    request_data = {req_id: [] for req_id in request_ids}

    # Pattern to capture each group starting with 'N=XX'
    group_pattern = r"N=(\d+)\n-+\n(.*?)(?=\n-+\nN=\d+|$)"
    groups = re.findall(group_pattern, log_data, re.DOTALL)

    for n_value, group_data in groups:
        if int(n_value) == request_num:
            # Parse each group if it matches the requested size
            iteration_pattern = r"Iteration: (\d+).*?Current Requests: \[(.*?)\]"
            iterations = re.findall(iteration_pattern, group_data, re.DOTALL)

            for iteration, requests in iterations:
                iteration = int(iteration)
                if requests:
                    # Process each request
                    requests = requests.strip('[]').split('}, {')
                    for req in requests:
                        # Clean up and extract details from each request
                        req = req.replace('{', '').replace('}', '')
                        req_id_match = re.search(r"'id': '(\d+)'", req)
                        tokens_match = re.search(r"'tokens': (\d+)", req)
                        if req_id_match and tokens_match:
                            req_id = req_id_match.group(1)
                            tokens = int(tokens_match.group(1))
                            if req_id in request_ids:
                                request_data[req_id].append((iteration, tokens))

    return request_data

def parse_request_num(filename, request_num):
    with open(filename, 'r') as file:
        log_data = file.read()
    # Pattern to capture each group starting with 'N=XX'
    group_pattern = r"N=(\d+)\n-+\n(.*?)(?=\n-+\nN=\d+|$)"
    groups = re.findall(group_pattern, log_data, re.DOTALL)
    for n_value, group_data in groups:
        if int(n_value) == request_num:
            iteration_pattern = r"Iteration: (\d+), Time:.*?Current Requests: \[(.*?)\]"
            iterations = re.findall(iteration_pattern, group_data, re.DOTALL)
            current_request_counts = []

            for iteration, requests in iterations:
                current_requests = requests.split('}, {') if requests else []
                current_request_counts.append(len(current_requests))

    return current_request_counts   

def plot_request_data(request_data, filename):
    plt.figure(figsize=(10, 6))

    # Set global font size
    plt.rcParams.update({'font.size': 14})

    for req_id, data in request_data.items():
        iterations, tokens = zip(*data)

        # Increase line width for plot lines
        plt.plot(iterations, tokens, label=f'Request {req_id}', linewidth=2)

    plt.xlabel('Iteration')
    plt.ylabel('Remaining Tokens')
    plt.title(filename.split('.png')[0])
    plt.grid(True)

    # Increase font size for legend
    plt.legend(fontsize='large')
    plt.savefig(filename)

def plot_request_counts(request_counts, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(request_counts)+1), request_counts, marker='o', linestyle='-', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Number of Pending Requests')
    plt.title('Pending Requests per Iteration')
    plt.grid(True)
    plt.savefig(filename)

'''
logfile = 'offline solver.log'  # Replace with your log file path
request_ids = ['10', '12', '3', '4', '5', '6', '7', '8', '9']  # Replace with your list of request IDs
request_num = 90  
request_data = parse_request_data(logfile, request_ids, request_num)
plot_request_data(request_data, 'tokenviz_'+logfile.split('.log')[0]+'.png')'''

logfile = 'random.log'
request_counts = parse_request_num(logfile, 300)
plot_request_counts(request_counts, 'pendingviz.png')

