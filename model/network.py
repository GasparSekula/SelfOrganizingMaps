import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from itertools import product
from sklearn.metrics import silhouette_score, davies_bouldin_score
from model.distance import GaussianDistance, Distance, MexicanHatDistance
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

PALETTE = 'viridis'

class KohonenNetwork:
    """Kohonen Self-Organizing Map (SOM) implementation."""
    
    def __init__(self, width: int, height: int, input_dim: int, hexagonal: bool = False, data: np.ndarray = None) -> None:
        """
        Initialize the KohonenNetwork.

        Args:
            width (int): The width of the SOM grid.
            height (int): The height of the SOM grid.
            input_dim (int): The dimensionality of the input data.
            hexagonal (bool, optional): Whether to use a hexagonal grid. Defaults to False.
        """
        self.width = width
        self.height = height
        self.input_dim = input_dim
        self.hexagonal_grid = hexagonal
        
        self._init_weights(data)
        self.grid = np.arange(width * height).reshape((width, height))
        
        self._init_map_coords()
        
        self.history = []
    
    def _init_weights(self, data: np.ndarray = None) -> None:
        """
        Initialize the weights of the SOM.

        Args:
            data (np.ndarray, optional): If provided, initializes weights using the given data.
                                         Otherwise, initializes weights randomly.
        """
        if data is None:
            self.weights = np.random.normal(size=(self.width, self.height, self.input_dim))
        else:
            indices = np.random.choice(len(data), self.width * self.height, replace=False)
            self.weights = data[indices].reshape((self.width, self.height, self.input_dim))
            
    def _init_map_coords(self) -> None:
        """
        Initialize the map coordinates for the SOM.

        This method sets up the coordinates of the neurons in the SOM grid. If the grid is hexagonal,
        the coordinates are adjusted to reflect the hexagonal layout. Otherwise, a standard rectangular
        grid is used.
        """
        self.numbered_matrix_fields = np.unravel_index(np.arange(self.width * self.height)\
                                                        .reshape(self.width, self.height),\
                                                        (self.width, self.height))
        self.map_coords_xy = copy.deepcopy(self.numbered_matrix_fields)
        if self.hexagonal_grid:
            cord_y = self.map_coords_xy[0]
            cord_y = cord_y.astype(float)
            cord_y = cord_y * np.sqrt(3)
            cord_y[:,(np.arange(self.height)*2)[np.arange(self.height)*2 < self.height]] += np.sqrt(3)/2

            cord_x = self.map_coords_xy[1]
            cord_x = cord_x.astype(float)
            cord_x = cord_x * 3/2

            self.map_coords_xy = (cord_y,cord_x)

    def _closest_to_sample(self, sample: np.ndarray) -> tuple:
        """
        Find the closest neuron to the given sample.

        Args:
            sample (np.ndarray): The input sample.

        Returns:
            tuple: The indices of the closest neuron.
        """
        distances = np.linalg.norm(sample - self.weights, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)
    
    def _get_distances(self, closest_to_sample_idx: tuple) -> np.ndarray:
        """
        Calculate distances from the closest neuron to all other neurons.

        Args:
            closest_to_sample_idx (tuple): Indices of the closest neuron.

        Returns:
            np.ndarray: Array of distances.
        """
        i, j = closest_to_sample_idx
        x0, y0 = self.map_coords_xy[0][i,j], self.map_coords_xy[1][i,j]
        dx = np.abs(self.map_coords_xy[0] - x0)
        dy = np.abs(self.map_coords_xy[1] - y0)
        return np.sqrt(dx**2 + dy**2)
        
    
    def train(self, data: np.ndarray, epochs: int = 100, neighbourhood_scaler: float = 0.5, learning_rate: float = 0.001, distance: Distance = GaussianDistance(1.0)) -> None:
        """
        Train the SOM using the input data.

        Args:
            data (np.ndarray): The input data for training.
            epochs (int, optional): Number of training epochs. Defaults to 100.
            neighbourhood_scaler (float, optional): Scaling factor for the neighborhood function. Defaults to 0.5.
            learning_rate (float, optional): Initial learning rate. Defaults to 0.001.
            distance (Distance, optional): Distance function to use. Defaults to GaussianDistance(1.0).
        """
        for epoch in range(epochs):
            lr = learning_rate * np.exp(-epoch / epochs)
            sigma = max(neighbourhood_scaler * np.exp(-epoch / epochs), 0.01)
            data_shuffled = np.random.permutation(data)
            
            for sample in data_shuffled:
                closest_to_sample_idx = self._closest_to_sample(sample)
                distances = self._get_distances(closest_to_sample_idx)
                neighbourhood_values = distance(distances / sigma)
                
                for i in range(self.width):
                    for j in range(self.height):                  
                        self.weights[i, j] += lr * neighbourhood_values[i, j] * (sample - self.weights[i, j])
                
            self.history.append(self.weights.copy())
    
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict the cluster indices for the given data.

        Args:
            data (np.ndarray): The input data to predict.

        Returns:
            np.ndarray: An array of cluster indices corresponding to the input data.
        """
        clusters = np.zeros((len(data), 2), dtype=int)
        for i, sample in enumerate(data):
            clusters[i] = self._closest_to_sample(sample)
        return clusters
    
    def predict_labels(self, data: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the given data based on the trained SOM.

        Args:
            data (np.ndarray): The input data to predict.

        Returns:
            np.ndarray: An array of predicted labels corresponding to the input data.
        """
        cluster_coords = self.predict(data)
        return self.grid[cluster_coords[:, 0], cluster_coords[:, 1]]
    
    def plot_map(self, data, classes, title=None, ax=None, symbols=True, labels=None) -> None:
        """
        Visualizes the Kohonen Network map with data points and their respective classes.

        Args:
            data (np.ndarray): The input data to be visualized on the map.
            classes (np.ndarray): The class labels corresponding to the input data.
            title (str, optional): The title of the plot. Defaults to None.
            ax (matplotlib.axes.Axes, optional): The axis to plot on. Defaults to None.
            symbols (bool, optional): Whether to display symbols for data points. Defaults to True.
            labels (list, optional): A list of labels for the classes. Defaults to None.

        Returns:
            None
        """
        max_dim = max(self.width, self.height)
        ax_original = ax
        if ax_original is None:
            fig, ax = plt.subplots(figsize=(self.width/max_dim*6,self.height/max_dim*6))
        else: 
            ax = ax_original
        num_classes = len(np.unique(classes))
        cmap = cm.get_cmap(PALETTE, num_classes)  
        norm = mcolors.Normalize(vmin=0, vmax=num_classes - 1)

        coords_x = self.map_coords_xy[0].astype(float)
        coords_y = self.map_coords_xy[1].astype(float)
        
        if self.hexagonal_grid:
            num_vertices = 6
            radius = 1
            orientation = np.radians(0)
            axis_limits = [np.min(coords_x) - 0.5 * np.sqrt(3), 
                           np.max(coords_x) + 1.5 * np.sqrt(3), 
                           np.min(coords_y) - 0.5,
                           np.max(coords_y) + 2.5]
        else:
            num_vertices = 4
            radius = np.sqrt(2)/2
            orientation = np.radians(45)
            axis_limits = [0, 
                           np.max(coords_x) + 2,
                           0,
                           np.max(coords_y) + 2]

        for x, y in zip(coords_x.flatten() + 1, coords_y.flatten() + 1):
            ax.add_patch(
                RegularPolygon(
                    (x, y),
                    numVertices=num_vertices,
                    radius=radius,
                    orientation=orientation,
                    facecolor="white",
                    alpha=1,
                    edgecolor='k'
                )
            )
        winning_count = np.zeros((self.width,self.height,len(np.unique(classes))))

        for i, data_row in enumerate(data):
            w = self._closest_to_sample(data_row)
            winning_count[w[0],w[1],classes[i]] += 1

            if symbols:
                epsilon_x = np.random.uniform(-0.2, 0.2)
                epsilon_y = np.random.uniform(-0.2, 0.2)
                ax.scatter(
                    coords_x[w[0], w[1]] + 1 + epsilon_x,
                    coords_y[w[0], w[1]] + 1 + epsilon_y,
                    s=20,
                    c=[cmap(norm(classes[i]))],
                    edgecolors='black',
                    linewidths=0.5,
                    alpha=0.8
                )

            ax.add_patch(
                RegularPolygon(
                    (coords_x[w[0],w[1]] + 1, coords_y[w[0],w[1]] + 1),
                    numVertices = num_vertices,
                    radius = radius,             
                    orientation = orientation,             
                    facecolor = cmap(norm(classes[i])),
                    alpha = 0.05,
                    edgecolor = 'k'
                )
            )

        if labels is not None:
            winning_class = np.argmax(winning_count, axis=2)
            for i in range(self.width):
                for j in range(self.height):
                    plt.text(self.map_coords_xy[0][i,j]+1,self.map_coords_xy[1][i,j]+1,
                            labels[winning_class[i,j]],
                            ha='center', va='center',fontsize=12)
        class_labels = [f"Class {i}" for i in range(num_classes)]
        class_colors = [cmap(norm(i)) for i in range(num_classes)]
        legend_handles = [mpatches.Patch(color=class_colors[i], label=class_labels[i]) for i in range(num_classes)]
        ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.15, 1), 
                  fontsize=6)

        ax.axis(axis_limits)
        
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title("Data represented by the Kohonen Network")
        if ax_original is None:
            plt.show()
        else:
            return 
        
    def plot_neurons_class_piecharts(self, data, classes, title=None, ax=None) -> None:
        """
        Plots a pie chart in each neuron representing the percentage of usage by each class.

        Args:
            data (np.ndarray): The input data.
            classes (np.ndarray): The class labels for the data.
            title (str, optional): The title of the plot. Defaults to None.
            ax (matplotlib.axes.Axes, optional): The axis to plot on. Defaults to None.
        """
        num_classes = len(np.unique(classes))
        class_counts = np.zeros((self.height, self.width, num_classes), dtype=int)
        for i, sample in enumerate(data):
            closest_idx = self._closest_to_sample(sample)
            class_counts[closest_idx[0], closest_idx[1], classes[i]] += 1

        ax_original = ax
        if ax_original is None:
            fig, ax = plt.subplots(figsize=(2*self.width, 2*self.height))
        else:
            ax = ax_original

        cmap = cm.get_cmap(PALETTE, len(np.unique(classes)))
        norm = mcolors.Normalize(vmin=0, vmax=num_classes - 1)
        
        coords_x = self.map_coords_xy[0].astype(float)
        coords_y = self.map_coords_xy[1].astype(float)
        
        def plot_pie(ax, ratios, x, y, size=2.0, zoom=1.0):
            total = np.sum(ratios)
            if total == 0:
                return
            fig, pie_ax = plt.subplots(figsize=(1,1))
            pie_ax.pie(ratios, colors=[cmap(i) for i in range(num_classes)], radius=size)
            pie_ax.axis('equal')
            fig.canvas.draw()
            
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            
            imagebox = OffsetImage(image, zoom=zoom)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False)
            ax.add_artist(ab)
        
        if self.hexagonal_grid:
            radius = 1
            orientation = np.radians(0)
            for i in range(self.width):
                for j in range(self.height):
                    x = coords_x[i, j]
                    y = coords_y[i, j]
                    ax.add_patch(RegularPolygon(
                        (x, y), numVertices=6, radius=radius,
                        orientation=orientation, edgecolor='k', facecolor='white', alpha=0.5
                    ))
                    plot_pie(ax, class_counts[i, j], x, y, zoom=0.75)
            ax.set_xlim(np.min(coords_x) - 1, np.max(coords_x) + 1.5)
            ax.set_ylim(np.min(coords_y) - 1, np.max(coords_y) + 1.5)
            ax.tick_params(axis='x', labelsize=16)
            ax.tick_params(axis='y', labelsize=16)
            ax.set_aspect('equal')

        else:
            for i in range(self.width):
                for j in range(self.height):
                    x = i + 0.5
                    y = j + 0.5
                    ax.add_patch(plt.Rectangle((i, j), 1, 1, edgecolor='k', facecolor='white', alpha=0.5))
                    plot_pie(ax, class_counts[i, j], x, y, zoom=1.0)
            ax.set_xlim(0, self.width)
            ax.set_ylim(0, self.height)
            ax.set_xticks(np.arange(self.width + 1))
            ax.set_yticks(np.arange(self.height + 1))
            ax.tick_params(axis='x', labelsize=16)
            ax.tick_params(axis='y', labelsize=16)
            ax.set_aspect('equal')
            ax.grid(True)

        class_labels = [f"Class {i}" for i in range(len(np.unique(classes)))]
        class_colors = [cmap(i) for i in range(len(class_labels))]
        legend_handles = [mpatches.Patch(color=class_colors[i], label=class_labels[i]) for i in range(len(class_labels))]
        ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.15,1), fontsize=16)
        
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title("Class distribution in each neuron", fontsize=24)

        if ax_original is None:
            plt.show() 
          
            
    def plot_neurons_class_labels(self, data, classes, title=None, ax=None) -> None:
        """
        Plots the most common class label in each neuron.

        Args:
            data (np.ndarray): The input data.
            classes (np.ndarray): The class labels for the data.
            title (str, optional): The title of the plot. Defaults to None.
            ax (matplotlib.axes.Axes, optional): The axis to plot on. Defaults to None.

        Returns:
            None
        """
        class_counts = np.zeros((self.height, self.width, len(np.unique(classes))), dtype=int)
        for i, sample in enumerate(data):
            closest_idx = self._closest_to_sample(sample)
            class_counts[closest_idx[0], closest_idx[1], classes[i]] += 1

        ax_original = ax
        if ax_original is None:
            fig, ax = plt.subplots(figsize=(self.width, self.height))
        else:
            ax = ax_original

        cmap = cm.get_cmap(PALETTE, len(np.unique(classes)))
        norm = mcolors.Normalize(vmin=0, vmax=len(np.unique(classes)) - 1)
        
        coords_x = self.map_coords_xy[0].astype(float)
        coords_y = self.map_coords_xy[1].astype(float)
        
        if self.hexagonal_grid:
            radius = 1
            orientation = np.radians(0)
            for i in range(self.width):
                for j in range(self.height):
                    if np.sum(class_counts[i, j]) == 0:
                        ax.add_patch(RegularPolygon(
                            (coords_x[i, j], coords_y[i, j]),
                            numVertices=6,
                            radius=radius,
                            orientation=orientation,
                            facecolor='white',
                            edgecolor='k',
                            alpha=0.5
                        ))
                        continue
                    most_common_class = np.argmax(class_counts[i, j])
                    x, y = coords_x[i, j], coords_y[i, j]
                    ax.add_patch(RegularPolygon(
                        (x, y),
                        numVertices=6,
                        radius=radius,
                        orientation=orientation,
                        facecolor=cmap(norm(most_common_class)),
                        edgecolor='k',
                        alpha=0.8
                    ))
                    ax.text(x, y, str(most_common_class),
                            ha='center', va='center', fontsize=17, color='black')
            ax.set_aspect('equal')
            ax.set_xlim(np.min(coords_x) - 1, np.max(coords_x) + 1.5)
            ax.set_ylim(np.min(coords_y) - 1, np.max(coords_y) + 1.5)
            
        else:
            for i in range(self.width):
                for j in range(self.height):
                    if np.sum(class_counts[i, j]) == 0:
                        continue
                    most_common_class = np.argmax(class_counts[i, j])
                    ax.add_patch(plt.Rectangle(
                        (i, j), 1, 1,
                        facecolor=cmap(norm(most_common_class)),
                        edgecolor='k',
                        alpha=0.8
                    ))
                    ax.text(i + 0.5, j + 0.5, str(most_common_class),
                            ha='center', va='center', fontsize=17, color='black')
            ax.set_aspect('equal')
            ax.set_xlim(0, self.width)
            ax.set_ylim(0, self.height)
            ax.set_xticks(np.arange(self.width + 1))
            ax.set_yticks(np.arange(self.height + 1))
            ax.grid(True)
            

        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title("Most common class in each neuron")
        if ax_original is None:
            plt.show()
            
    def plot_kohonen_grid_progress(
        self, X: np.ndarray, labels: np.ndarray, sigma: float = None, cube: bool = False
    ) -> None:
        """
        Plots the progress of the Kohonen network grid during training.

        Args:
            X (np.ndarray): The input data.
            labels (np.ndarray): The labels.
            sigma (float, optional): The width of the neighborhood function. Defaults to None.
            cube (bool, optional): Whether to plot in 3D or 2D. Defaults to False.
        """
        epochs = len(self.history)
        weights_0 = self.history[0]
        weights_last = self.history[-1]

        fig = plt.figure(figsize=(14, 6))
        for idx, (weights, epoch) in enumerate(
            zip([weights_0, weights_last], [0, epochs])
        ):
            if cube:
                ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
                scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, alpha=0.3, cmap=PALETTE)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
            else:
                ax = fig.add_subplot(1, 2, idx + 1)
                scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, alpha=0.4, cmap=PALETTE)
                ax.set_xlabel("x")
                ax.set_ylabel("y")

            draw_connections = self.height * self.width < 15
            for i, j in product(range(self.width), range(self.height)):
                neuron = weights[i, j]
                if cube:
                    ax.scatter(neuron[0], neuron[1], neuron[2], color="red")
                    if draw_connections:
                        if i + 1 < self.width:
                            neighbor = weights[i + 1, j]
                            ax.plot3D(
                                [neuron[0], neighbor[0]],
                                [neuron[1], neighbor[1]],
                                [neuron[2], neighbor[2]],
                                "k-",
                            )
                        if j + 1 < self.height:
                            neighbor = weights[i, j + 1]
                            ax.plot3D(
                                [neuron[0], neighbor[0]],
                                [neuron[1], neighbor[1]],
                                [neuron[2], neighbor[2]],
                                "k-",
                            )
                else:
                    ax.plot(neuron[0], neuron[1], "ro")
                    if draw_connections:
                        if i + 1 < self.width:
                            neighbor = weights[i + 1, j]
                            ax.plot(
                                [neuron[0], neighbor[0]], [neuron[1], neighbor[1]], "k-"
                            )
                        if j + 1 < self.height:
                            neighbor = weights[i, j + 1]
                            ax.plot(
                                [neuron[0], neighbor[0]], [neuron[1], neighbor[1]], "k-"
                            )

            ax.set_title(
                f"Kohonen Network - Epoch {epoch if epoch > 0 else 0}"
                + (f" ($\\sigma$ = {sigma})" if sigma else "")
            )

            if not cube:
                legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
                ax.add_artist(legend1)

        plt.tight_layout()
        plt.show()
    
    def plot_clusters(self, data, labels) -> None:
        """
        This method visualizes the clustering results of the Kohonen Network.

        It creates a side-by-side comparison of the true class labels and the predicted class labels
        for the input data. The true classes are displayed on the left, while the predicted classes
        are displayed on the right.

        Args:
            data (np.ndarray): The input data to be visualized.
            labels (np.ndarray): The true class labels corresponding to the input data.

        Returns:
            None
        """
        labels_pred = self.predict_labels(data)
        fig, ax = plt.subplots(figsize=(20, 10))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.scatter(data[:, 0], data[:, 1], c=labels, cmap=PALETTE, alpha=0.6)
        ax1.set_title('True Classes', fontsize=20)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(data[:, 0], data[:, 1], c=labels_pred, cmap=PALETTE, alpha=0.6)
        ax2.set_title('Predicted Classes', fontsize=20)
        ax2.set_xlabel('Feature 1')
        ax2.set_ylabel('Feature 2')
        
    
    def metrics(self, data, labels, verbose = True) -> dict:
        """
        Evaluate the Kohonen Network using various clustering metrics.

        This method calculates and returns the number of unique classes in the data,
        the number of unique classes predicted by the Kohonen Network, the silhouette score,
        and the Davies-Bouldin score. These metrics provide insights into the quality of the
        clustering performed by the network.

        Args:
            data (np.ndarray): The input data to evaluate.
            labels (np.ndarray): The true class labels corresponding to the input data.
            verbose (bool, optional): Whether to print detailed metrics. Defaults to True.

        Returns:
            dict: A dictionary containing the following keys:
            - "num_classes": The number of unique classes in the data.
            - "num_classes_pred": The number of unique classes predicted by the network.
            - "silhouette_score": The silhouette score of the clustering.
            - "davies_bouldin_score": The Davies-Bouldin score of the clustering.
        """
        classes_predicted = self.predict_labels(data)
        
        num_classes_pred = np.unique(classes_predicted)
        num_classes = np.unique(labels)
        
        silhouette = silhouette_score(data, classes_predicted)
        davies_bouldin = davies_bouldin_score(data, classes_predicted)
        
        if verbose:
            print(f"Number of neurons in the Kohonen network: {self.width * self.height}")
            print("***")
            print(f"Number of classes in the data: {len(num_classes)}")
            print(f"Number of classes predicted by the Kohonen network: {len(num_classes_pred)}")
            print("***")
            print(f"Silhouette score: {silhouette:.4f}")
            print(f"Davies-Bouldin score: {davies_bouldin:.4f}")
        
        return {"num_classes": len(num_classes), 
                "num_classes_pred": len(num_classes_pred), 
                "silhouette_score": silhouette, 
                "davies_bouldin_score": davies_bouldin}