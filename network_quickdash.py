import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
import json
import numpy as np
import igraph as ig

app = dash.Dash(__name__)

# Read the precomputed Sankey data
with open('sankey_data.json', 'r') as f:
    sankey_data = json.load(f)

min_year = min(map(int, sankey_data.keys()))
max_year = max(map(int, sankey_data.keys()))

def create_network(connections_df, selected_N):
    # Ensure the 'Count' column is of numeric type
    connections_df['Count'] = pd.to_numeric(connections_df['Count'], errors='coerce')
    
    # Drop rows with non-numeric 'Count' values
    connections_df = connections_df.dropna(subset=['Count'])
    
    top_connections = connections_df.nlargest(selected_N, 'Count')
    
    # Create a directed graph
    G = ig.Graph(directed=True)
    
    all_nodes = pd.concat([top_connections['Source'], top_connections['Target']]).unique()
    G.add_vertices(all_nodes)
    
    for index, row in top_connections.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Count'])
   
    # Compute the layout of the graph
    layout = G.layout('kk')  # Kamada-Kawai layout
    
    # Extract node positions from the layout
    node_x = [pos[0] for pos in layout.coords]
    node_y = [pos[1] for pos in layout.coords]
    
    # Compute the size of nodes based on the total connection count
    node_sizes = []
    node_labels = []
    for node in G.vs:
        in_edges = G.es.select(_target_in=[node.index])
        out_edges = G.es.select(_source_in=[node.index])
        total_weight = sum(in_edges['weight']) + sum(out_edges['weight'])
        node_sizes.append(np.sqrt(total_weight) * 0.7)  # Adjust size multiplier as needed
        node_labels.append(node['name'])

    label_trace = go.Scatter(
        x=node_x, y=node_y,
        text=node_labels,
        textposition="top center",
        mode='text',
        hoverinfo='none'
    )
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            size=node_sizes,  # Set node sizes based on total connection count
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )
    
    node_text = []
    for node in G.vs:
        node_text.append(f'{node["name"]}<br># of connections: {G.degree(node.index)}')
    
    node_trace.text = node_text
    
    # Normalize edge weights to range [0, 1]
    edge_weights = np.array(G.es['weight'])
    edge_weight_range = max(edge_weights) - min(edge_weights)
    normalized_weights = (edge_weights - min(edge_weights)) / (edge_weight_range if edge_weight_range != 0 else 1)

    min_alpha = 0.1  # Set minimum alpha for visibility
    alpha_range = 0.9  # Set the range of alpha values
    
    traces = []  # List to store all the traces
    
    for edge, weight in zip(G.es, normalized_weights):
        source_idx, target_idx = edge.tuple
        edge_x = [layout.coords[source_idx][0], layout.coords[target_idx][0]]
        edge_y = [layout.coords[source_idx][1], layout.coords[target_idx][1]]
        
        alpha = min_alpha + alpha_range * weight  # Ensure a minimum alpha for visibility
        thickness = 1 + 9 * weight  # Adjust thickness based on weight, e.g., from 1 to 10
        
        # Create a trace for each edge
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=thickness, color=f'rgba(0, 0, 0, {alpha})'),
            hoverinfo='text',
            text=f'Connections: {edge["weight"]}',  # Hover text
            mode='lines'
        )
        
        traces.append(edge_trace)

    hover_x = []
    hover_y = []
    hover_text = []
    for edge, weight in zip(G.es, edge_weights):
        source_idx, target_idx = edge.tuple
        x0, y0 = layout.coords[source_idx]
        x1, y1 = layout.coords[target_idx]
        hover_x.append((x0 + x1) / 2)
        hover_y.append((y0 + y1) / 2)
        hover_text.append(f'Connections: {edge["weight"]}')

    # Creating a trace for edge hover points
    hover_trace = go.Scatter(
        x=hover_x, y=hover_y,
        text=hover_text,
        mode='markers',
        hoverinfo='text',
        marker=dict(color='rgba(0,0,0,0)', size=5),
    )

    # Create a Plotly figure
    fig = go.Figure(data=traces + [node_trace, label_trace, hover_trace],  # Add all edge traces and the node trace
             layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0,l=0,r=0,t=0),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    return fig

@app.callback(
    Output('software-connections', 'figure'),
    [Input('n-selector', 'value'),
     Input('year-slider', 'value')]
)
def update_graph(selected_N, year_range):
    combined_connections = pd.DataFrame(columns=['Source', 'Target', 'Count'])
    
    for year in range(year_range[0], year_range[1] + 1):
        year_str = str(year)
        if year_str in sankey_data:
            year_connections = pd.DataFrame(sankey_data[year_str])
            combined_connections = pd.concat([combined_connections, year_connections])
    
    combined_connections = combined_connections.groupby(['Source', 'Target'], as_index=False).sum()
    return create_network(combined_connections, selected_N)

# Dash app layout
app.layout = html.Div([
    html.Div([
        html.Div([
            html.H6('N Most Cited Software:', style={'marginBottom': 5, 'marginTop': 0}),
            dcc.Dropdown(
                id='n-selector',
                options=[
                    {'label': 'Top 5', 'value': 5},
                    {'label': 'Top 10', 'value': 10},
                    {'label': 'Top 25', 'value': 25},
                    {'label': 'Top 50', 'value': 50},
                    {'label': 'Top 100', 'value': 100},
                ],
                value=50  # default value
            )
        ], style={'width': '49%', 'display': 'inline-block', 'padding': '10px', 'boxShadow': '0px 0px 5px #ccc', 'borderRadius': '5px'}),
    ], style={'marginBottom': '10px'}),

    html.Div([
        dcc.RangeSlider(
            id='year-slider',
            min=1970,
            max=2021,
            step=1,
            marks={i: str(i) for i in range(1970, 2021 + 1, 5)},
            value=[1995, max_year]  # default value
        )
    ], style={'padding': '10px', 'boxShadow': '0px 0px 5px #ccc', 'borderRadius': '5px', 'marginBottom': '20px'}),

    dcc.Graph(
        id='software-connections',
        style={'height': '70vh'}  # Set the height of the graph
    ),
], style={'padding': '10px', 'height': '100vh', 'margin': '0'})

if __name__ == '__main__':
    app.run_server(debug=False, port=8052)
