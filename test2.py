import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
from sklearn.cluster import KMeans
import random


# Load the stylists and relationships data
stylists_data = pd.read_csv("stylists_data.csv")
relationships_data = pd.read_csv("following_relationships.csv")


# Function to create the base network
def create_base_network():
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes
    for _, row in stylists_data.iterrows():
        G.add_node(
            row['Name'], 
            followers=row['Followers'], 
            salon=row['Salon Name'],
            popular_post=row['Most Popular Post Text'],
            likes=row['Likes']
        )

    # Add edges
    for _, row in relationships_data.iterrows():
        G.add_edge(row['Follower'], row['Followed'])

    return G


# Function to visualize the network in PyVis
def visualize_network(G, influencers=[], clustering=None, selected_account=None):
    # Create a PyVis Network
    net = Network(height="750px", width="100%", directed=True)
    net.from_nx(G)

    # Highlight influencers (if provided) by changing node size or color
    for node in net.nodes:
        if node["id"] in influencers:
            node["color"] = "red"
            node["size"] = 20

    # Cluster nodes (if clustering is provided)
    if clustering:
        for node in net.nodes:
            node["group"] = clustering.get(node["id"], 0)  # Assign cluster group

    # Highlight selected account (if provided)
    if selected_account:
        for node in net.nodes:
            if node["id"] == selected_account:
                node["color"] = "green"
                node["size"] = 30
                break

    # Customize general appearance
    net.set_options("""
    const options = {
      "nodes": {
        "shape": "dot",
        "size": 15,
        "font": {
          "size": 15
        }
      },
      "edges": {
        "color": {"inherit": true},
        "smooth": false
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -40000,
          "centralGravity": 0.3
        },
        "minVelocity": 0.75
      }
    }
    """)

    return net


# Function to calculate influencers (PageRank)
def calculate_influencers(G, top_n=10):
    pagerank_scores = nx.pagerank(G)
    top_influencers = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    influencer_names = [node for node, _ in top_influencers]
    return influencer_names


# Function to perform clustering
def cluster_nodes(G):
    adjacency_matrix = nx.to_numpy_array(G)
    kmeans = KMeans(n_clusters=5, random_state=42).fit(adjacency_matrix)
    clusters = [int(label) for label in kmeans.labels_]
    node_cluster_map = {node: cluster for node, cluster in zip(G.nodes(), clusters)}
    return node_cluster_map


# Main Streamlit App
st.title("Instagram Stylist Network Analysis")

# Create the base network graph
G = create_base_network()

# Sidebar options
st.sidebar.header("Options")
selected_node = st.sidebar.selectbox("Select an Account (Node):", ["None"] + list(stylists_data['Name']))
show_influencers = st.sidebar.checkbox("Highlight Key Influencers", value=True)
show_clustering = st.sidebar.checkbox("Show Communities (Clusters)", value=True)

# Prepare data for each feature
influencers = calculate_influencers(G) if show_influencers else []
clustering = cluster_nodes(G) if show_clustering else None

# Display network graph
if selected_node == "None":
    selected_node = None  # Reset to avoid issues

net = visualize_network(G, influencers=influencers, clustering=clustering, selected_account=selected_node)
net.save_graph("pyvis_network.html")
st.components.v1.html(open("pyvis_network.html", "r").read(), height=800)

# Display account details (if selected)
if selected_node:
    st.sidebar.subheader("Account Details")
    account_data = stylists_data[stylists_data['Name'] == selected_node].iloc[0]
    st.sidebar.write(f"**Name:** {account_data['Name']}")
    st.sidebar.write(f"**Salon:** {account_data['Salon Name']}")
    st.sidebar.write(f"**Followers:** {account_data['Followers']}")
    st.sidebar.write(f"**Most Popular Post:** {account_data['Most Popular Post Text']}")
    st.sidebar.write(f"**Likes:** {account_data['Likes']}")

# Display additional information in the main page
if influencers:
    st.subheader("Top 10 Influencers in the Network")
    st.write(", ".join(influencers))

if show_clustering and clustering:
    st.subheader("Community Clustering Summary:")
    cluster_counts = pd.Series(list(clustering.values())).value_counts()
    for cluster, count in cluster_counts.items():
        st.write(f"Cluster {cluster}: {count} nodes")