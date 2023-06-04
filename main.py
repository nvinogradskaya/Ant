import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QGridLayout, QMessageBox, QGraphicsScene, QGraphicsView, QGraphicsSimpleTextItem
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QFont
from PyQt6.QtCore import Qt
import math
import random


class Ant:
    def __init__(self, start_point):
        self.start_point = start_point
        self.visited = {start_point} # множество посещенных вершин
        self.path = [start_point] # путь муравья
        self.path_length = 0

    def visit(self, point):
        self.visited.add(point)
        self.path.append(point)

    def calculate_path_length(self, graph):
        self.path_length = 0
        num_points = len(graph)
        for i in range(num_points - 1):
            start = self.path[i]
            end = self.path[i + 1]
            self.path_length += graph[start][end]
        self.path_length += graph[self.path[-1]][self.start_point]

    def select_next_point(self, graph, pheromone, alpha, beta):
        current_point = self.path[-1]
        unvisited = [point for point in range(len(graph)) if point not in self.visited]
        if len(unvisited) == 0:
            return self.start_point
        p = []
        total = 0
        for point in unvisited:
            pheromone_value = pheromone[current_point][point]
            heuristic_value = 1 / graph[current_point][point]
            probability = pheromone_value ** alpha * heuristic_value ** beta
            p.append(probability)
            total += probability
        r = random.uniform(0, total)
        cumulative = 0
        for i, point in enumerate(unvisited):
            cumulative += p[i]
            if cumulative >= r:
                self.visit(point)
                self.path_length += graph[current_point][point]
                return point
        return None

    def reset(self, num_points):
        self.visited = {self.start_point}
        self.path = [self.start_point]
        self.path_length = 0

class TSP(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Решение задачи коммивояжера")
        self.setGeometry(500, 500, 1000, 800)
        self.size_label = QLabel("Количество вершин графа:")
        self.size_input = QLineEdit()
        self.generate_button = QPushButton("Ввод матрицы")
        self.generate_button.clicked.connect(self.generate_matrix)

        self.matrix_layout = QGridLayout()
        self.solve_button = QPushButton("Старт")
        self.solve_button.clicked.connect(self.start)

        self.result_label = QLabel()
        self.graph_widget = GraphWidget()

        layout = QVBoxLayout()
        layout.addWidget(self.size_label)
        layout.addWidget(self.size_input)
        layout.addWidget(self.generate_button)
        layout.addLayout(self.matrix_layout)
        layout.addWidget(self.solve_button)
        layout.addWidget(self.result_label)
        layout.addWidget(self.graph_widget)
        self.setLayout(layout)

    def generate_matrix(self):
        num_points = int(self.size_input.text())
        self.clear_matrix()
        for row in range(num_points):
            for col in range(num_points):
                cell_label = QLabel(f"Из пункта {chr(row + 65)} в пункт {chr(col + 65)}:")
                cell_input = QLineEdit()
                self.matrix_layout.addWidget(cell_label, row, col * 2)
                self.matrix_layout.addWidget(cell_input, row, col * 2 + 1)

    def clear_matrix(self):
        while self.matrix_layout.count():
            item = self.matrix_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def get_adj_matrix(self, num_points):
        adjacency_matrix = []
        for row in range(num_points):
            matrix_row = []
            for col in range(num_points):
                cell_input = self.matrix_layout.itemAtPosition(row, col * 2 + 1).widget()
                value = cell_input.text().strip()
                if value.isdigit():
                    matrix_row.append(int(value))
                else:
                    matrix_row.append(0)
            adjacency_matrix.append(matrix_row)
        return adjacency_matrix

    def parse_adj_matrix(self, adjacency_matrix, num_points):
        graph = [[math.inf for _ in range(num_points)] for _ in range(num_points)]
        for i in range(num_points):
            for j in range(num_points):
                graph[i][j] = adjacency_matrix[i][j]
        return graph

    def create_pheromone_matrix(self, num_points, initial_pheromone):
        pheromone = [[initial_pheromone for _ in range(num_points)] for _ in range(num_points)]
        return pheromone

    def update_pheromone(self, pheromone, ants, evaporation_rate):
        num_points = len(pheromone)
        for i in range(num_points):
            for j in range(num_points):
                pheromone[i][j] *= (1 - evaporation_rate)
                for ant in ants:
                    pheromone[i][j] += ant.path_length

    def start(self):
        num_points = int(self.size_input.text())
        adjacency_matrix = self.get_adj_matrix(num_points)
        try:
            graph = self.parse_adj_matrix(adjacency_matrix, num_points)
            num_ants = num_points
            ants = [Ant(start_point=i) for i in range(num_ants)]
            pheromone = self.create_pheromone_matrix(num_points, initial_pheromone=0.1)
            num_iterations = 100
            alpha = 1.0
            beta = 2.0
            evaporation_rate = 0.5
            best_path = None
            best_length = float('inf')
            for _ in range(num_iterations):
                for ant in ants:
                    ant.reset(num_points)
                    while len(ant.visited) < num_points:
                        ant.select_next_point(graph, pheromone, alpha, beta)
                    ant.calculate_path_length(graph)
                self.update_pheromone(pheromone, ants, evaporation_rate)
                for ant in ants:
                    if ant.path_length < best_length:
                        best_length = ant.path_length
                        best_path = ant.path
            if best_path is not None:
                letter_path = [chr(point + 65) for point in best_path]
                letter_path.append(letter_path[0])
                self.result_label.setText(f"Длина кратчайшего гамильтонова цикла: {best_length}\n"
                                          f"Кратчайший гамильтонов цикл: {' --> '.join(letter_path)}")
                self.graph_widget.set_graph(graph, num_points, best_path)
            else:
                self.result_label.setText("Гамильтонова цикла не существует")
        except ValueError:
            QMessageBox.critical(self, "Ошибка при вводе матрицы смежности", "Ошибка при вводе матрицы смежности")

class GraphWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.view)
        self.points = {}
        self.edges = []

    def set_graph(self, adjacency_matrix, num_points, path):
        self.points = {}
        self.edges = []
        self.num_points = num_points
        self.path = path
        center_x = self.view.width() / 2
        center_y = self.view.height() / 2
        radius = min(center_x, center_y) * 0.8
        angle = 360 / self.num_points
        for i in range(self.num_points):
            x = center_x + radius * math.cos(math.radians(angle * i))
            y = center_y + radius * math.sin(math.radians(angle * i))
            self.points[i] = (x, y)

        for i in range(self.num_points):
            for j in range(self.num_points):
                if adjacency_matrix[i][j] != 0:
                    self.edges.append((i, j))
        self.draw_graph()

    def draw_graph(self):
        self.scene.clear()
        pen = QPen(QColor("black"), 1)
        for start, end in self.edges:
            start_x, start_y = self.points[start]
            end_x, end_y = self.points[end]
            self.scene.addLine(start_x, start_y, end_x, end_y, pen)

        point_radius = 8
        brush = QBrush(QColor("black"), Qt.BrushStyle.SolidPattern)
        for i, (x, y) in self.points.items():
            self.scene.addEllipse(x - point_radius, y - point_radius, 2 * point_radius, 2 * point_radius, pen, brush)

        label_font = QFont("Arial", 18)
        for i, (x, y) in self.points.items():
            label = QGraphicsSimpleTextItem(chr(ord('A') + i))
            label.setFont(label_font)
            label.setPos(x + point_radius, y + point_radius)
            label.setZValue(-1)
            self.scene.addItem(label)

        if self.path:
            path_pen = QPen(QColor("red"), 2)
            for i in range(len(self.path)):
                start_x, start_y = self.points[self.path[i]]
                end_x, end_y = self.points[self.path[(i + 1) % len(self.path)]]
                self.scene.addLine(start_x, start_y, end_x, end_y, path_pen)

        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


if __name__ == "__main__":
    app = QApplication([])
    window = TSP()
    window.show()
    sys.exit(app.exec())
