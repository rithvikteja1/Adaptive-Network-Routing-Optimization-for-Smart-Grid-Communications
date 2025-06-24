import string
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import threading
import time

random.seed(100)
np.random.seed(50)


def cal_distance(path):
    dis = 0
    for i in range(len(path) - 1):
        dis += D[path[i]][path[i + 1]]
    return dis


def plot_graph(adjacency_matrix, traffic, figure_title=None, paths=None, colors=None):
    adjacency_matrix = np.array(adjacency_matrix)
    rows, cols = np.where(adjacency_matrix > 0)
    edges = list(zip(rows.tolist(), cols.tolist()))
    values = [adjacency_matrix[i][j] for i, j in edges]
    weighted_edges = [(e[0], e[1], values[idx]) for idx, e in enumerate(edges)]
    
    # Clear the current figure
    plt.clf()
    plt.title(figure_title or "Multi-Packet Q-Learning")
    
    # Create the graph
    G = nx.Graph()
    G.add_weighted_edges_from(weighted_edges)
    labels = nx.get_edge_attributes(G, 'weight')
    pos = nx.kamada_kawai_layout(G)
    
    # Visualize traffic as node colors
    node_colors = [min(traffic[node] / max(1, max(traffic)), 1) for node in range(len(traffic))]
    nx.draw(G, pos, with_labels=True, font_size=15, node_color=node_colors, cmap=plt.cm.Blues)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=12)
    
    # Highlight paths with different colors
    if paths:
        for idx, path in enumerate(paths):
            nx.draw_networkx_edges(G, pos, edgelist=path, edge_color=colors[idx], width=2)
    
    plt.pause(0.3)  # Pause to allow the plot to update dynamically


def q_learning_multi_packet(start_states, target_states, num_epoch=200, gamma=0.8, epsilon=0.05, alpha=0.1, visualize=True):
    print("-" * 20)
    print("Q-learning with multiple packets begins ...")
    
    if len(start_states) != len(target_states):
        raise Exception("Start and target states must have the same length!")
    
    num_packets = len(start_states)
    q = np.zeros((num_nodes, num_nodes))  
    traffic = np.zeros(num_nodes)  # Track traffic at each node
    
    best_paths = [[] for _ in range(num_packets)]
    shortest_paths = [float('inf')] * num_packets
    colors = ['r', 'g', 'b']  # Distinct colors for packets
    
    plt.figure(figsize=(10, 8))
    
    total_delivered_packets = 0
    total_path_lengths = 0
    
    for epoch in range(1, num_epoch + 1):
        print(f"\nEpoch {epoch}")
        packet_current_states = start_states[:]
        epoch_paths = [[] for _ in range(num_packets)]
        epoch_lengths = [0] * num_packets
        successful_deliveries = 0
        
        while any(s != t for s, t in zip(packet_current_states, target_states)):
            for packet_idx in range(num_packets):
                s_cur = packet_current_states[packet_idx]
                t_cur = target_states[packet_idx]
                
                if s_cur == t_cur:  # Skip packets that reached their target
                    continue

                print(f"Packet {packet_idx + 1}: Current Node = {s_cur}, Target Node = {t_cur}")

                # Check if nodes are directly connected
                if D[s_cur][t_cur] > 0:
                    print(f"  Direct delivery: Packet {packet_idx + 1} moves from {s_cur} to {t_cur}")
                    epoch_paths[packet_idx].append((s_cur, t_cur))
                    epoch_lengths[packet_idx] += D[s_cur][t_cur]
                    traffic[s_cur] += 1
                    traffic[t_cur] += 1
                    packet_current_states[packet_idx] = t_cur
                    successful_deliveries += 1
                    continue

                # Choose next state using Q-learning
                potential_next_states = np.where(np.array(D[s_cur]) > 0)[0]
                if np.random.rand() < epsilon:
                    s_next = random.choice(potential_next_states)
                else:
                    rewards = -np.array(D[s_cur])[potential_next_states] - traffic[potential_next_states]
                    s_next = potential_next_states[np.argmax(rewards)]

                if s_next != t_cur:
                    print(f"  Deviation: Packet {packet_idx + 1} deviates to {s_next} (due to traffic).")

                # Update Q-table
                delta = -D[s_cur][s_next] - traffic[s_next] + gamma * np.max(q[s_next]) - q[s_cur][s_next]
                q[s_cur][s_next] += alpha * delta

                # Move packet
                traffic[s_cur] += 1
                packet_current_states[packet_idx] = s_next
                epoch_paths[packet_idx].append((s_cur, s_next))
                epoch_lengths[packet_idx] += D[s_cur][s_next]

                if s_next == t_cur:
                    print(f"  Packet {packet_idx + 1} reaches its target {t_cur}.")
                    successful_deliveries += 1
                    traffic[s_next] += 1
        
        # Update best paths
        for packet_idx in range(num_packets):
            if epoch_lengths[packet_idx] < shortest_paths[packet_idx]:
                shortest_paths[packet_idx] = epoch_lengths[packet_idx]
                best_paths[packet_idx] = epoch_paths[packet_idx]
        
        # Update total delivered packets and path lengths for metric calculation
        total_delivered_packets += successful_deliveries
        total_path_lengths += sum(epoch_lengths)
        
        if visualize:
            plot_graph(D, traffic, figure_title=f"Epoch {epoch}", paths=epoch_paths, colors=colors)
    
    plt.show()
    print("\nFinal Metrics:")
    # print(f"Total Delivered Packets: {total_delivered_packets}")
    print(f"Packet Delivery Ratio: {total_delivered_packets / (num_packets * num_epoch):.4f}")
    print(f"Average Path Length: {total_path_lengths / total_delivered_packets:.4f}")


if __name__ == '__main__':
    D = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
         [4, 0, 8, 0, 0, 0, 0, 11, 0],
         [0, 8, 0, 7, 0, 4, 0, 0, 3],
         [0, 0, 7, 0, 9, 14, 0, 0, 0],
         [0, 0, 0, 9, 0, 10, 0, 0, 0],
         [0, 0, 4, 14, 10, 0, 3, 0, 0],
         [0, 0, 0, 0, 0, 3, 0, 3, 4],
         [8, 11, 0, 0, 0, 0, 3, 0, 5],
         [0, 0, 3, 0, 0, 0, 4, 5, 0]]
    num_nodes = len(D)

    start_states = [3, 5, 7]
    target_states = [0, 1, 3]

    # Run Q-learning
    q_learning_multi_packet(start_states, target_states, num_epoch=100, gamma=0.8, epsilon=0.05, alpha=0.1, visualize=True)