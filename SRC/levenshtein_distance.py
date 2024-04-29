import copy
import networkx
import numpy as np
import matplotlib.pyplot as plt


class Pattern:
    def __init__(self, A, B):
        self.A = A
        self.B = B
        self.levenshtein_distance = -1
        self.levenshtein_distance_matrix = self.levenshteinDistance()
        self.node_pos = {}
        self.edge_labels = {}
        self.node_labels = {}
        self.node_color_map = []
        self.graph = self.build_graph()

    def levenshteinDistance(self):

        N, M = len(self.A), len(self.B)
        # Create an array of size NxM
        dp = [[0 for i in range(M + 1)] for j in range(N + 1)]

        # Base Case: When N = 0
        for j in range(M + 1):
            dp[0][j] = j
        # Base Case: When M = 0
        for i in range(N + 1):
            dp[i][0] = i
        # Transitions
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                if self.A[i - 1] == self.B[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],  # Insertion
                        dp[i][j - 1],  # Deletion
                        dp[i - 1][j - 1]  # Replacement
                    )
        self.levenshtein_distance = dp[-1][-1]
        return np.asarray(dp)

    def get_node_index_from_pos(self, pos):
        for node_index in self.node_pos:
            if self.node_pos[node_index] == pos:
                return node_index

    def add_path(self, g, shortest_paths):
        for path in shortest_paths:
            for node_index in range(len(path) - 1):
                g[path[node_index]][path[node_index + 1]]["color"] = "green"
        return g

    def node_paths_to_pos_paths(self, paths):
        pos_paths = []
        for path in paths:
            pos_path = []
            for move in path:
                pos_path.append(self.node_pos[move])
            pos_paths.append(pos_path)
        return pos_paths

    def plot_levenshtein_matrix_matrix(self):
        graph = copy.deepcopy(self.graph)
        edges = graph.edges()
        colors = [graph[u][v]['color'] for u, v in edges]

        invert_pos = {x: [self.node_pos[x][0], -self.node_pos[x][1]] for x in self.node_pos}
        labels = copy.deepcopy(self.node_labels)
        color_map = copy.deepcopy(self.node_color_map)
        self.add_letter_to_graph(graph, invert_pos, labels, color_map)

        networkx.draw(graph, node_color=color_map, pos=invert_pos, edge_color=colors,
                      width=[3 for x in range(len(colors))])
        networkx.draw_networkx_edge_labels(
            graph, pos=invert_pos,
            edge_labels=self.edge_labels,
            font_color='red'
        )

        networkx.draw_networkx_labels(graph, invert_pos, labels)
        plt.show()

    def add_letter_to_graph(self, g, node_pos, node_labels, node_color_map):
        for letter in range(len(self.A)):
            g.add_node(len(node_pos))
            node_pos[len(node_pos)] = [letter + 1, 0.3]
            node_labels[len(node_pos) - 1] = self.A[letter]
            node_color_map.append("blue")

        for letter in range(len(self.B)):
            g.add_node(len(node_pos))
            node_pos[len(node_pos)] = [-0.3, -1 - letter]
            node_labels[len(node_pos) - 1] = self.B[letter]
            node_color_map.append("blue")

    def plot_path(self):
        all_shortest_paths = list(
            networkx.all_shortest_paths(self.graph, source=0, target=len(self.node_pos) - 1, weight="weight"))
        graph = self.add_path(self.graph, all_shortest_paths)
        edges = graph.edges()
        colors = [graph[u][v]['color'] for u, v in edges]

        invert_pos = {x: [self.node_pos[x][0], -self.node_pos[x][1]] for x in self.node_pos}
        labels = copy.deepcopy(self.node_labels)
        color_map = copy.deepcopy(self.node_color_map)
        self.add_letter_to_graph(graph, invert_pos, labels, color_map)

        networkx.draw(graph, node_color=color_map, pos=invert_pos, edge_color=colors,
                      width=[3 for x in range(len(colors))])
        networkx.draw_networkx_edge_labels(
            graph, pos=invert_pos,
            edge_labels=self.edge_labels,
            font_color='red'
        )

        networkx.draw_networkx_labels(graph, invert_pos, labels)
        plt.show()

    def build_graph(self):
        g = networkx.DiGraph()
        node_index = 0
        for row in range(np.shape(self.levenshtein_distance_matrix)[0]):
            for col in range(np.shape(self.levenshtein_distance_matrix)[1]):
                g.add_node(node_index)
                self.node_pos[node_index] = [row, col]
                self.node_labels[node_index] = f'{self.levenshtein_distance_matrix[row, col]}'
                self.node_color_map.append("red")
                node_index += 1

        for row in range(np.shape(self.levenshtein_distance_matrix)[0]):
            for col in range(np.shape(self.levenshtein_distance_matrix)[1]):
                if row + 1 < np.shape(self.levenshtein_distance_matrix)[0]:
                    node_from = self.get_node_index_from_pos([row, col])
                    node_to = self.get_node_index_from_pos([row + 1, col])
                    w = self.levenshtein_distance_matrix[row + 1][col]
                    g.add_edge(node_from, node_to, weight=w, color="black")
                    key = (node_from, node_to)
                    self.edge_labels[key] = f"-{self.A[row]}"
                if col + 1 < np.shape(self.levenshtein_distance_matrix)[1]:
                    node_from = self.get_node_index_from_pos([row, col])
                    node_to = self.get_node_index_from_pos([row, col + 1])
                    g.add_edge(node_from, node_to, weight=self.levenshtein_distance_matrix[row][col + 1], color="black")
                    key = (node_from, node_to)
                    self.edge_labels[key] = f"+{self.B[col]}"
                if row + 1 < np.shape(self.levenshtein_distance_matrix)[0] and col + 1 < \
                        np.shape(self.levenshtein_distance_matrix)[1]:
                    node_from = self.get_node_index_from_pos([row, col])
                    node_to = self.get_node_index_from_pos([row + 1, col + 1])
                    w = 0
                    if self.levenshtein_distance_matrix[row + 1][col + 1] > self.levenshtein_distance_matrix[row][col]:
                        w = 0.5
                    g.add_edge(node_from, node_to, weight=self.levenshtein_distance_matrix[row + 1][col + 1] + w,
                               color="black")
                    key = (node_from, node_to)
                    self.edge_labels[key] = f"{self.A[row]}=>{self.B[col]}"

        return g

    def compute_error_patterns(self):
        all_shortest_paths = list(
            networkx.all_shortest_paths(self.graph, source=0, target=len(self.node_pos) - 1, weight="weight"))
        self.graph = self.add_path(self.graph, all_shortest_paths)
        pos_paths = self.node_paths_to_pos_paths(all_shortest_paths)
        pattern = self.get_patterns(pos_paths)
        return pattern

    def get_patterns(self, paths):
        """
        param paths: all paths from start to last letter
        :return: for each the list of transformations applied to arrive from sequence A to sequence B

        Note: a transformation can be: insertion, deletion, substitution. When both letters of sequence A and B
        at a given place are the same the letter is labelled as Correct (no transformation).
        """
        patterns = []
        for path in paths:
            pattern = []
            for index_move in range(1, len(path)):
                if path[index_move - 1][0] < path[index_move][0] and path[index_move - 1][1] == path[index_move][1]:
                    pattern.append(("Deletion", self.A[path[index_move][0] - 1]))
                elif path[index_move - 1][0] == path[index_move][0] and path[index_move - 1][1] < path[index_move][1]:
                    pattern.append(("Insertion", self.B[path[index_move][1] - 1]))
                elif path[index_move - 1][0] < path[index_move][0] and path[index_move - 1][1] < path[index_move][1] \
                        and path[index_move - 1][-1] < path[index_move][-1]:
                    if self.A[path[index_move][0] - 1] == self.B[path[index_move][1] - 1]:
                        pattern.append(("Correct", f'{self.A[path[index_move][0] - 1]}'))
                    else:
                        pattern.append(
                            ("Substitution", f'{self.A[path[index_move][0] - 1]}=>{self.B[path[index_move][1] - 1]}'))
                else:
                    print("Error!")
            patterns.append(pattern)

        return patterns
