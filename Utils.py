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
        
        src_prefix = '.'.join(row['srcIp'].split('.')[:2])
        dst_prefix = '.'.join(row['dstIp'].split('.')[:2])
        if src_prefix == dst_prefix:
            kmeans_df.loc[row['srcIp'], 'P2P'] = 1
    

    return kmeans_df
def filter_by_subnet_and_threshold(data, threshold=0.07):
    subnets = set(index.rsplit('.', 1)[0] for index in data.index)
    filtered_data = pd.DataFrame()
    for subnet in subnets:
        subnet_data = data[data.index.str.startswith(subnet)]
        if not subnet_data.empty:
            sum_by_port = subnet_data.drop(columns='Cluster').sum(axis=0)
            total_entries = len(subnet_data)
            if (sum_by_port / total_entries).max() > threshold:
                filtered_data = pd.concat([filtered_data, subnet_data])
    return filtered_data
    

def perform_kmeans (kmeans_data,num_clusters):

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(kmeans_data)
    kmeans_data['Cluster'] = kmeans.labels_
    silhouette_avg = silhouette_score(kmeans_data.drop(columns='Cluster'), kmeans_data['Cluster'])
    db_index = davies_bouldin_score(kmeans_data.drop(columns='Cluster'), kmeans_data['Cluster'])
    print(f'K: {num_clusters}, Silhouette Score: {silhouette_avg}, Davies-Bouldin Index: {db_index}')
    return kmeans_data

def filter_patterns_in_cluster(cluster):
    threshold_percentage = 0.04
    total_lines = len(cluster)
    threshold_value = total_lines * threshold_percentage
    patterns = cluster.apply(lambda row: tuple(row[:-1] == 1), axis=1)
    pattern_counts = patterns.value_counts()
    patterns_to_keep = pattern_counts[pattern_counts >= threshold_value].index
    filtered_cluster = cluster[patterns.isin(patterns_to_keep)]
    return filtered_cluster, total_lines- len(filtered_cluster)

def plot_data( kmeans_data, num_clusters, with_noise):
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(kmeans_data.drop(columns='Cluster'))
    pca_df = pd.DataFrame(pca_result, columns=['PCA Component 1', 'PCA Component 2'], index=kmeans_data.index)
    pca_df['Cluster'] = kmeans_data['Cluster'].astype(str)
    # Plotly
    fig_pca = px.scatter(pca_df, x='PCA Component 1', y='PCA Component 2', color='Cluster',
                         title=f'PCA of Clusters with K={num_clusters}')
    fig_pca.show()

    cmap = plt.colormaps['tab20']
    colors = [cmap(i) for i in range(num_clusters)]
    fig, axes = plt.subplots(math.ceil(num_clusters / 4), 4, figsize=(15, 10))

    for cluster in range(num_clusters):
        row = cluster // 4
        col = cluster % 4
        cluster_data = kmeans_data[kmeans_data['Cluster'] == cluster]
        if cluster_data.empty:
            continue
        if with_noise==False:
            cluster_data,dropped_data =  filter_patterns_in_cluster(cluster_data)
        
        first_octet = cluster_data.index.str.split('.').str[0]
        first_octet_percentage = first_octet.value_counts(normalize=True) * 100
        first_octet_count = first_octet.value_counts()
        
        title_ports = cluster_data.drop(columns='Cluster').loc[:, cluster_data.drop(columns='Cluster').sum(axis=0) > 0].sum(axis=0).sort_values(ascending=False).index.tolist()
        if with_noise:
            title = f'Cluster {cluster}: ' + ', '.join(title_ports)
        else:
            title = f'Cluster {cluster}: ' + ', '.join(title_ports) + f' dropped {dropped_data*100/ len(cluster_data):.1f}'

        # Plot 
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

    
    plt.tight_layout()
    plt.show()
    
