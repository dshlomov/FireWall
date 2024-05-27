import random
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import plotly.express as px
import math

def generate_network_data(subnets, dest_ports, connections_per_subnet=10000, designated_port_ratio=0.8,additional_port_ratio=0.3):
    
    def generate_ip(subnet):
        first_octet = subnet.split('.')[0]
        second_octet = subnet.split('.')[1]
        return f"{first_octet}.{second_octet}.{random.randint(1, 254)}.{random.randint(1, 254)}"
      
    def generate_dstip():
        return f"{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
    
    def generate_port():
        return random.randint(1024, 65535)
    
    def choose_protocol():
        return random.choice(['TCP', 'UDP'])
    
    data = []
    designated_port_count = int(connections_per_subnet * designated_port_ratio)
    random_port_count = connections_per_subnet - designated_port_count
    
    for subnet, designated_port in zip(subnets, dest_ports):
        for _ in range(designated_port_count):
            if subnet == "22.22." and random.random() < 0.3:
                src_ip = generate_ip("22.22.")
                dst_ip = generate_ip("22.22.")
                #print (dst_ip + " " + src_ip)
            else:
                src_ip = generate_ip(subnet)
                dst_ip = generate_dstip()
                
            src_port = generate_port()
            protocol = choose_protocol()
            #dst_ip = generate_dstip()
            dst_port = designated_port
            data.append((src_ip, src_port, protocol, dst_ip, dst_port))
            #print ((src_ip, src_port, protocol, dst_ip, dst_port))  
            
            if designated_port == 21:
                if random.random() < additional_port_ratio:
                    data.append((src_ip, generate_port(), protocol, dst_ip, 443))
            if designated_port == 80:
                if random.random() < additional_port_ratio:
                    data.append((src_ip, generate_port(), protocol, dst_ip, 443))
        
        for _ in range(random_port_count):
            src_ip = generate_ip(subnet)
            src_port = generate_port()
            protocol = choose_protocol()
            dst_ip = generate_dstip()
            dst_port = random.choice(dest_ports)
            data.append((src_ip, src_port, protocol, dst_ip, dst_port))
    
    df = pd.DataFrame(data, columns=["srcIp", "srcPort", "protocol", "dstIp", "dstPort"])
    return df

def prepare_kmeans_data(df, target_ports):
    target_ports = [str(port) for port in target_ports]
    unique_src_ips = df['srcIp'].unique()
    
    kmeans_df = pd.DataFrame(0, index=unique_src_ips, columns=target_ports + ['P2P'])
    
    for _, row in df.iterrows():
        if str(row['dstPort']) in target_ports:
            kmeans_df.loc[row['srcIp'], str(row['dstPort'])] = 1
        
        # Check if the srcIp prefix matches the dstIp prefix and set P2P column
        src_prefix = '.'.join(row['srcIp'].split('.')[:2])
        dst_prefix = '.'.join(row['dstIp'].split('.')[:2])
        if src_prefix == dst_prefix:
            kmeans_df.loc[row['srcIp'], 'P2P'] = 1
    

    return kmeans_df
def perform_kmeans_and_plot_pca(kmeans_data, num_clusters):

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(kmeans_data)
    
    # Add the cluster labels to the DataFrame
    kmeans_data['Cluster'] = kmeans.labels_
    
    silhouette_avg = silhouette_score(kmeans_data.drop(columns='Cluster'), kmeans_data['Cluster'])
    db_index = davies_bouldin_score(kmeans_data.drop(columns='Cluster'), kmeans_data['Cluster'])
    print(f'K: {num_clusters}, Silhouette Score: {silhouette_avg}, Davies-Bouldin Index: {db_index}')
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(kmeans_data.drop(columns='Cluster'))
    
    # Create a DataFrame for PCA results
    pca_df = pd.DataFrame(pca_result, columns=['PCA Component 1', 'PCA Component 2'], index=kmeans_data.index)
    pca_df['Cluster'] = kmeans_data['Cluster'].astype(str)
    
    # Interactive PCA plot with Plotly
    fig_pca = px.scatter(pca_df, x='PCA Component 1', y='PCA Component 2', color='Cluster',
                         title=f'PCA of Clusters with K={num_clusters}')
    fig_pca.show()
    
    # Define colors using the updated method
    cmap = plt.colormaps['tab20']
    colors = [cmap(i) for i in range(num_clusters)]

    # Create a figure with subplots
    fig, axes = plt.subplots(math.ceil(num_clusters / 4), 4, figsize=(15, 10))

    for cluster in range(num_clusters):
        # Calculate the position in the subplot grid
        row = cluster // 4
        col = cluster % 4

        # Filter the data for the specific cluster
        cluster_data = kmeans_data[kmeans_data['Cluster'] == cluster]

        # Check if the cluster data is empty
        if cluster_data.empty:
            continue

        # Extract the first octet from the srcIp
        first_octet = cluster_data.index.str.split('.').str[0]

        # Calculate the percentage of each unique first octet
        first_octet_percentage = first_octet.value_counts(normalize=True) * 100
        
        # Calculate the count of each unique first octet
        first_octet_count = first_octet.value_counts()
        
        # Generate the title based on the destination ports that have a value of 1
       # title_ports = cluster_data.drop(columns='Cluster').columns[(cluster_data.drop(columns='Cluster').sum(axis=0) > 0)].tolist()
        #title_ports = cluster_data.drop(columns='Cluster').sum(axis=0).sort_values(ascending=False).index.tolist()
        title_ports = cluster_data.drop(columns='Cluster').loc[:, cluster_data.drop(columns='Cluster').sum(axis=0) > 0].sum(axis=0).sort_values(ascending=False).index.tolist()


        title = f'Cluster {cluster}: ' + ', '.join(title_ports)

        # Plot the percentages and counts using dual axes
        ax = axes[row, col]
        ax2 = ax.twinx()
        
        first_octet_percentage.sort_index().plot(kind='bar', alpha=0.7, color=colors[cluster], ax=ax, position=0, width=0.4)
        first_octet_count.sort_index().plot(kind='bar', alpha=0.3, color='gray', ax=ax2, position=1, width=0.4)
        
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('Percentage', fontsize=8)
        ax.set_xlabel('First Octet', fontsize=8)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax2.set_ylabel('Count', fontsize=8)
        ax2.tick_params(axis='y', labelsize=8)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()