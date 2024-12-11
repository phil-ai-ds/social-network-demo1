#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
from sklearn.cluster import KMeans
import numpy as np

# Load the stylists and relationships data
stylists_data = pd.read_csv("stylists_data.csv")
relationships_data = pd.read_csv("following_relationships.csv")

# Function to create and visualize the network graph
def create_pyvis_network():
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes (stylists)
    for index, row in stylists_data.iterrows():
        G.add_node(
            row["Name"], 
            title=f"{row['Name']}<br>Salon: {row['Salon Name']}<br>Followers: {row['Followers']}",
            followers=row['Followers']
        )

    # Add edges (follower relationships)
    for index, row in relationships_data.iterrows():
        G.add_edge(row["Follower"], row["Followed"])

    # Create PyVis network
    net = Network(height="700px", width="100%", notebook=False, directed=True)
    net.from_nx(G)
    net.set_options("""
    const options = {
      "interaction": {"hover": true},
      "nodes": {"color": {"hover": {"border": "black"}}},
      "physics": {"barnesHut": {"gravitationalConstant": -10000}},
      "edges": {"smooth": false}
    }
    """)
    return net, G

# Function to calculate social network influence
def calculate_influence(G):
    # Use NetworkX "pagerank" to score each node based on their influence
    pagerank_scores = nx.pagerank(G)
    return pagerank_scores

# Perform clustering (Optional: Color by community)
def cluster_nodes(G):
    # Convert nodes to a structured 2D array by adjacency matrix for clustering
    adjacency_matrix = nx.to_numpy_array(G)
    # Perform KMeans clustering (e.g., 5 communities for simplicity)
    kmeans = KMeans(n_clusters=5, random_state=42).fit(adjacency_matrix)
    clusters = [int(label) for label in kmeans.labels_]  # Convert labels to Python int
    node_cluster_map = {node: cluster for node, cluster in zip(G.nodes(), clusters)}
    return node_cluster_map

# Step 1: Set up the Streamlit sidebar and select options
st.sidebar.title("Instagram Network Analysis")
action = st.sidebar.selectbox(
    "Choose an Action:", 
    ["Network Overview", "Node Details", "Top Influencers", "Community Clustering"]
)

# Step 2: Network Overview
if action == "Network Overview":
    st.title("Instagram Connection Network")
    with st.spinner("Loading network..."):
        net, G = create_pyvis_network()
        net.save_graph("pyvis_network.html")
        st.components.v1.html(open("pyvis_network.html", "r").read(), height=800)

# Step 3: Node Details
elif action == "Node Details":
    st.title("Node Details")
    selected_node = st.selectbox("Select a Node:", stylists_data["Name"])
    selected_node_data = stylists_data[stylists_data["Name"] == selected_node].iloc[0]

    st.subheader(f"Details for: {selected_node}")
    st.write(f"**Salon Name:** {selected_node_data['Salon Name']}")
    st.write(f"**Followers:** {selected_node_data['Followers']}")
    st.write(f"**Most Popular Post:** {selected_node_data['Most Popular Post Text']}")
    st.write(f"**Likes on Popular Post:** {selected_node_data['Likes']}")

# Step 4: Top Influencers
elif action == "Top Influencers":
    st.title("Top 10 Influencers in the Network")
    with st.spinner("Calculating influence..."):
        net, G = create_pyvis_network()
        pagerank_scores = calculate_influence(G)
        sorted_influencers = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        st.subheader("Top 10 Influential Nodes:")
        for rank, (node, score) in enumerate(sorted_influencers, 1):
            node_data = stylists_data[stylists_data["Name"] == node].iloc[0]
            st.write(f"{rank}. **{node}** (Salon: {node_data['Salon Name']} | Followers: {node_data['Followers']} | Score: {score:.4f})")

# Step 5: Community Clustering
elif action == "Community Clustering":
    st.title("Community Clustering")
    with st.spinner("Clustering and visualizing the network..."):
        # Create the graph and use clustering
        net, G = create_pyvis_network()
        node_cluster_map = cluster_nodes(G)

        # Assign clusters and color-code nodes by clusters (convert to int for JSON serialization)
        for node in net.nodes:
            node_id = node["id"]  # Properly fetch the node ID from the PyVis node dict
            node["group"] = int(node_cluster_map.get(node_id, 0))  # Convert cluster label to int

        # Save and render PyVis clustered network
        net.save_graph("pyvis_clustered_network.html")
        st.components.v1.html(open("pyvis_clustered_network.html", "r").read(), height=800)

        # Reports the distribution of clusters
        cluster_counts = pd.Series(list(node_cluster_map.values())).value_counts()
        st.subheader("Cluster Distribution:")
        for cluster_id, count in cluster_counts.items():
            st.write(f"Cluster {cluster_id}: {count} members")


# In[ ]:




