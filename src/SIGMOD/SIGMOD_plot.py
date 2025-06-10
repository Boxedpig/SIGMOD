import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from typing import Dict, Tuple
import networkx as nx

from adjustText import adjust_text

import os


def draw_spatial_pie(proportion, spatial_location, colors=None, radius=None, seed=12345,figsize=(7,7),file_name=None):
    
    res_spatial = pd.DataFrame(proportion)
    location = pd.DataFrame(spatial_location)

    res_spatial = res_spatial[sorted(res_spatial.columns)]

    if not all(res_spatial.index == location.index):
        raise ValueError("The rownames of proportion data do not match with the rownames of spatial location data.")

    color_candidate = [
        "#1e77b4", "#ff7d0b", "#ceaaa3", "#2c9f2c", "#babc22", "#d52828", "#9267bc", "#8b544c", "#e277c1", 
        "#d42728", "#adc6e8", "#97df89", "#fe9795", "#4381bd", "#f2941f", "#5aa43a", "#cc4d2e", "#9f83c8", 
        "#91675a", "#da8ec8", "#929292", "#c3c237", "#b4e0ea", "#bacceb", "#f7c685", "#dcf0d0", "#f4a99f", 
        "#c8bad8", "#F56867", "#FEB915", "#C798EE", "#59BE86", "#7495D3", "#D1D1D1", "#6D1A9C", "#15821E", 
        "#3A84E6", "#997273", "#787878", "#DB4C6C", "#9E7A7A", "#554236", "#AF5F3C", "#93796C", "#F9BD3F", 
        "#DAB370", "#877F6C", "#268785", "#f4f1de", "#e07a5f", "#3d405b", "#81b29a", "#f2cc8f", "#a8dadc", 
        "#f1faee", "#f08080"
    ]

    if colors is None:
        np.random.seed(seed)
        if res_spatial.shape[1] > len(color_candidate):
            from matplotlib import cm
            colors = cm.get_cmap("tab20", res_spatial.shape[1]).colors
        else:
            colors = np.random.choice(color_candidate, res_spatial.shape[1], replace=False)
    else:
        colors = colors

    data = pd.concat([res_spatial, location], axis=1)
    ct_select = res_spatial.columns.tolist()

    if radius is None:
        width = data['x'].max() - data['x'].min()
        height = data['y'].max() - data['y'].min()
        radius = np.sqrt((width * height) / len(data)) / np.pi * 0.85

    fig, ax = plt.subplots(figsize=figsize)
    for i, row in data.iterrows():
        x, y = row["x"], row["y"]
        ratios = row[ct_select].values
        total = ratios.sum()
        start_angle = 0
        for j, r in enumerate(ratios):
            if total == 0:
                continue
            angle = r / total * 360
            wedge = plt.matplotlib.patches.Wedge(center=(x, y), r=radius,
                                                  theta1=start_angle, theta2=start_angle + angle,
                                                  facecolor=colors[j], edgecolor='none')
            ax.add_patch(wedge)
            start_angle += angle

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(ct_select))]
    ax.legend(handles, ct_select, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              ncol=min(len(ct_select), 5), fontsize='small', title='Cell Type')

    ax.set_xlim(data['x'].min() - radius, data['x'].max() + radius)
    ax.set_ylim(data['y'].min() - radius, data['y'].max() + radius)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')
    plt.tight_layout()

    if file_name is not None:
        dire = './Figure'
        os.makedirs(dire, exist_ok=True)
        plt.savefig(os.path.join(dire, file_name), bbox_inches='tight')
    plt.show()



def plot_circle_interactions(
    interaction: pd.DataFrame,
    node_color_map: Dict[str,str] = None,
    edge_cmap: str = "viridis",
    min_width: float = 1.0,
    max_width: float = 8.0,
    threshold: float = 0.0,
    node_size: float = 600.0,
    font_size: float = 9.0,
    figsize: Tuple[int,int] = (8,8),
    arc_rad: float = 0.15,
    file_name: str = None,
):
    """
    Draw a circular plot of cell–cell interactions with:
      - each node in its own color
      - edges whose width and color encode interaction strength

    Parameters
    ----------
    interaction : pd.DataFrame
        Square (n×n) interaction matrix. Rows = sources, cols = targets.
    node_color_map : dict, optional
        Mapping {node_name: color} for each celltype. If None, 
        a default seaborn 'tab20' palette is used.
    edge_cmap : str
        Name of a matplotlib colormap for edges.
    min_width, max_width : float
        Range of line widths for edges (mapped from smallest→largest weight).
    threshold : float
        Minimum weight for drawing an edge.
    figsize : (w,h)
        Figure size in inches.
    arc_rad : float
        Curvature of the edge arcs. 0 = straight lines.
    """
    # --- 1) Build graph ---
    G = nx.DiGraph()
    nodes = list(interaction.index)
    G.add_nodes_from(nodes)
    # collect edges above threshold
    weights = []
    for src in nodes:
        for tgt in nodes:
            w = interaction.at[src, tgt]
            if w > threshold:
                G.add_edge(src, tgt, weight=w)
                weights.append(w)
    if not weights:
        raise ValueError("No edges above the threshold!")

    # --- 2) Choose node colors ---
    if node_color_map is None:
        palette = sns.color_palette("tab20", len(nodes))
        node_color_map = {node: palette[i] for i,node in enumerate(nodes)}
    node_colors = [node_color_map[n] for n in nodes]
    
    # --- 3) Circular layout ---
    pos = nx.circular_layout(G, scale=1)

    # --- 4) Draw nodes ---
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_size,
        node_color=node_colors,
        edgecolors="black",
        linewidths=0.5
    )

    # --- 5) Prepare edge styling ---
    all_weights = np.array([d["weight"] for _,_,d in G.edges(data=True)])
    wmin, wmax = all_weights.min(), all_weights.max()
    norm = Normalize(vmin=wmin, vmax=wmax)
    cmap = cm.get_cmap(edge_cmap)

    edge_widths = []
    edge_colors = []
    for u, v, data in G.edges(data=True):
        w = data["weight"]
        # normalize [0,1]
        wn = norm(w)
        # line width
        edge_widths.append(min_width + wn*(max_width-min_width))
        # edge color
        edge_colors.append(node_color_map[u])

    # --- 6) Draw edges as curved arcs ---
    nx.draw_networkx_edges(
        G, pos,
        edgelist=list(G.edges()),
        width=edge_widths,
        edge_color=edge_colors,
        connectionstyle=f"arc3,rad={arc_rad}",
        arrowsize=8
    )

    # --- 7) Draw labels ---
    nx.draw_networkx_labels(
        G, pos,
        font_size=font_size,
        font_family="DeJavu Serif"
    )

    # --- 8) Colorbar for edge weights ---
    # sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    # sm.set_array([])
    # cbar = plt.colorbar(sm, fraction=0.05, pad=0.02)
    # cbar.set_label("Interaction weight")

    plt.axis("off")
    plt.title("Module–module interaction circle plot", pad=20)
    plt.tight_layout()
    if file_name is not None:
        plt.savefig(file_name,bbox_inches='tight')
    plt.show()


def plot_gene_interaction_network(df, lr_genes, node_color_map, figsize=(4, 9), weight_threshold=0.01, 
                                 node_size=100, edge_width_scale=5, title="Gene-Gene Interaction Network",
                                 save_path=None, dpi=300):
    """
    Plot a directed gene-gene interaction network using NetworkX with bipartite layout.
    Edge colors match the sender node's color.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with rows as responder genes and columns as sender genes, containing interaction weights.
    lr_genes : dict
        Dictionary mapping group names to lists of genes for coloring nodes.
    node_color_map : dict
        Dictionary mapping group prefixes to colors for node visualization.
    figsize : tuple
        Figure size for the plot (width, height).
    weight_threshold : float
        Minimum weight for adding an edge (default: 0.01).
    node_size : int
        Size of nodes in the plot (default: 100).
    edge_width_scale : float
        Scaling factor for edge widths (default: 5).
    title : str
        Title of the plot (default: "Gene-Gene Interaction Network").
    """
    # Initialize a directed graph
    G = nx.DiGraph()

    # Add nodes for each gene/cell type
    responder_genes = df.index.tolist()  # Rows as responder genes
    sender_genes = df.columns.tolist()   # Columns as sender genes

    # Assign colors to nodes based on test_genes
    node_colors = {}
    for group, genes_list in lr_genes.items():
        for gene in genes_list:
            if gene in responder_genes or gene in sender_genes:
                node_colors[gene] = node_color_map[group.split("_")[0]]

    # Add nodes for all genes
    for gene in responder_genes:
        G.add_node(gene, type="gene", color=node_colors.get(gene, "lightgrey"))
    for gene in sender_genes:
        G.add_node(gene, type="gene", color=node_colors.get(gene, "lightgrey"))

    # Add edges with weights from the matrix
    for gene in responder_genes:
        for sg in sender_genes:
            weight = df.loc[gene, sg]
            if weight > weight_threshold:
                G.add_edge(sg, gene, weight=weight)

    # Normalize edge weights for visualization
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    min_weight, max_weight = min(edge_weights), max(edge_weights)
    norm_edge_weights = [(weight - min_weight) / (max_weight - min_weight) * edge_width_scale + 0.5 
                        for weight in edge_weights]

    # Assign edge colors to match sender node colors
    edge_colors = [G.nodes[u]['color'] for u, v in G.edges()]

    # Extract node colors for visualization
    node_color_list = [G.nodes[node]['color'] for node in G.nodes()]

    # Create figure
    plt.figure(figsize=figsize)

    # Use bipartite layout
    pos = nx.bipartite_layout(G, sender_genes)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_color_list, node_size=node_size, 
                          alpha=0.6, edgecolors="black")

    # Draw edges with colors matching sender nodes
    nx.draw_networkx_edges(G, pos, width=norm_edge_weights, edge_color=edge_colors, 
                          alpha=0.8, arrows=True)

    # Add labels with offset
    texts = []
    for node, (x, y) in pos.items():
        offset_x = 0.05 if x > 0 else -0.4
        offset_y = -0.005 if y > 0 else -0.005
        label = node.split('_')[-1]
        text = plt.text(x + offset_x, y + offset_y, label, fontsize=8, fontweight='bold')
        texts.append(text)

    # Finalize plot
    plt.title(title)
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, format=save_path.split('.')[-1], dpi=dpi, bbox_inches='tight')

    plt.show()