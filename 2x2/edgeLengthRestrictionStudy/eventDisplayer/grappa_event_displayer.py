#!/usr/bin/env python3
# coding: utf-8

import sys
import yaml
import numpy as np
import plotly.express as px
from multiprocessing import Pool

sys.path.append('lartpc_mlreco3d')

from mlreco.main_funcs import process_config, prepare
from mlreco.main_funcs import apply_event_filter

from mlreco.utils.inference import get_inference_cfg
from mlreco.utils.gnn.cluster import form_clusters
from mlreco.utils.gnn.cluster import get_cluster_label
from mlreco.utils.gnn.evaluation import node_assignment_score

from mlreco.visualization.gnn import network_topology
from mlreco.visualization.points import scatter_points

from mlreco.visualization.plotly_layouts import plotly_layout3d
#from mlreco.visualization.plotly_layouts import high_contrast_colorscale

from plotly import graph_objs as go

from matplotlib import colors as mcolors

from scipy.special import softmax

#File Paths

fifty_shower_paths = {"cfg_path" : "../50Restriction/grappa_shower.cfg",
                       "dataset_path" : "/sdf/data/neutrino/kterao/dunend_train_prod/prod2x2_v0_1_larnd2supera/combined/test.root",
                       "model_path" : "../50Restriction/weights/grappa_shower/50_edge_length/snapshot-11499.ckpt",
                       "dir" : "50Restriction"}
fifty_track_paths = {"cfg_path" : "../50Restriction/grappa_track.cfg",
                       "dataset_path" : "/sdf/data/neutrino/kterao/dunend_train_prod/prod2x2_v0_1_larnd2supera/combined/test.root",
                       "model_path" : "../50Restriction/weights/grappa_track/50_edge_length/snapshot-10999.ckpt",
                       "dir" : "50Restriction"}
fifty_inter_paths = {"cfg_path" : "../50Restriction/grappa_inter.cfg",
                       "dataset_path" : "/sdf/data/neutrino/kterao/dunend_train_prod/prod2x2_v0_1_larnd2supera/combined/test.root",
                       "model_path" : "../50Restriction/weights/grappa_inter/50_edge_length/snapshot-8499.ckpt",
                       "dir" : "50Restriction"}

no_shower_paths = {"cfg_path" : "../noRestriction/grappa_shower.cfg",
                       "dataset_path" : "/sdf/data/neutrino/kterao/dunend_train_prod/prod2x2_v0_1_larnd2supera/combined/test.root",
                       "model_path" : "../noRestriction/weights/grappa_shower/no_edge_length_restriction/snapshot-9999.ckpt",
                       "dir" : "noRestriction"}
no_track_paths = {"cfg_path" : "../noRestriction/grappa_track.cfg",
                       "dataset_path" : "/sdf/data/neutrino/kterao/dunend_train_prod/prod2x2_v0_1_larnd2supera/combined/test.root",
                       "model_path" : "../noRestriction/weights/grappa_track/no_edge_length_restriction/snapshot-999.ckpt",
                       "dir" : "noRestriction"}
no_inter_paths = {"cfg_path" : "../noRestriction/grappa_inter.cfg",
                       "dataset_path" : "/sdf/data/neutrino/kterao/dunend_train_prod/prod2x2_v0_1_larnd2supera/combined/test.root",
                       "model_path" : "../noRestriction/weights/grappa_inter/no_edge_length_restriction/snapshot-4499.ckpt",
                       "dir" : "noRestriction"}

def prep_cfg(cfg_path, dataset_path, model_path):
    cfg = get_inference_cfg(cfg_path=cfg_path, batch_size=1, num_workers=0)

    # Override the dataset
    cfg['iotool']['dataset']['data_keys'] = [dataset_path]

    # Pick weights
    cfg['trainval']['model_path'] = model_path

    # Turn on unwrapper
    cfg['trainval']['unwrapper'] = 'unwrap'

    return cfg

def add_particle_parser(cfg):
    cfg['iotool']['dataset']['schema']['particles'] ={
        'parser': 'parse_particles',
        'args': {
            'particle_event': 'particle_pcluster', 
            'cluster_event': 'cluster3d_pcluster'
        }
    }

    return cfg

def instantiate_handlers(cfg, event_id):
    # Pre-process configuration
    process_config(cfg)

    hs = prepare(cfg)

    print('Number of events in set:', len(hs.data_io))

    apply_event_filter(hs, [event_id])

    return  hs.trainer.forward(hs.data_io_iter)

def set_layout():
    layout = plotly_layout3d()
    layout['legend']['title'] = 'Legend'
    layout['scene']['camera'] = dict(eye=dict(x=1.5,y=0.75,z=1.5), up=dict(x=0,y=1,z=0), center=dict(x=0, y=-0.25, z=0.1))
    layout['scene']['aspectmode'] = 'manual'
    layout['scene']['aspectratio'] = dict(x=1, y=1, z=1)

    return layout

def create_grappa_shower_truth(event, output, graph):
    # Get track labels
    cluster_label = event['input_data'][0]

    # Draw fragments
    graph += network_topology(event['input_data'][0][:,1:4], output['clusts'][0], clust_labels=np.arange(len(output['clusts'][0])), markersize=2)
    graph[-1]['name'] = 'Shower Fragments'

    # Draw shower truth info
    shower_shapes = get_cluster_label(event['input_data'][0], output['clusts'][0], -1)
    shower_labels = np.unique(get_cluster_label(event['input_data'][0], output['clusts'][0], 6), return_inverse=True)[-1]
    shower_primary_labels = np.unique(get_cluster_label(event['input_data'][0], output['clusts'][0], 10), return_inverse=True)[-1]
    graph += network_topology(event['input_data'][0][:,1:4], output['clusts'][0], clust_labels=shower_shapes, markersize=2)
    graph[-1]['name'] = 'Shower Semantics'
    graph += network_topology(event['input_data'][0][:,1:4], output['clusts'][0], clust_labels=shower_labels, markersize=2)
    graph[-1]['name'] = 'Shower Labels'
    graph += network_topology(event['input_data'][0][:,1:4], output['clusts'][0], clust_labels=shower_primary_labels, colorscale='Portland', cmin=0, cmax=1, markersize=2)
    graph[-1]['name'] = 'Shower Primary Labels'

    return graph

def create_grappa_shower_prediction(event, output, graph):
    # Get track labels
    cluster_label = event['input_data'][0]
    # Draw shower predictions
    shower_preds = np.unique(node_assignment_score(output['edge_index'][0], output['edge_pred'][0], len(output['clusts'][0])), return_inverse=True)[-1]
    shower_primary_preds = softmax(output['node_pred'][0], axis=1)[:,-1]
    shower_edge_mask = np.argmax(output['edge_pred'][0], axis=-1).astype(bool)
    shower_edge_index = output['edge_index'][0][shower_edge_mask]
    graph += network_topology(event['input_data'][0][:,1:4], output['clusts'][0], shower_edge_index, clust_labels=shower_preds, markersize=2)
    graph[-2]['name'] = 'Shower Predictions'
    graph[-1]['name'] = 'Shower Edge Predictions'
    graph += network_topology(event['input_data'][0][:,1:4], output['clusts'][0], clust_labels=shower_primary_preds, colorscale='Portland', cmin=0, cmax=1, markersize=2)
    graph[-1]['name'] = 'Shower Primary Predictions'

    print('Edge accuracy', output['edge_accuracy'])
    print('Node accuracy', output['node_accuracy'])

    return graph

def create_grappa_track_truth(event, output, graph):
    # Get track labels
    cluster_label = event['input_data'][0]

    # Draw fragments
    graph = []
    graph += network_topology(event['input_data'][0][:,1:4], output['clusts'][0], clust_labels=np.arange(len(output['clusts'][0])), markersize=2)
    graph[-1]['name'] = 'Track Fragments'

    # Draw track truth info
    track_labels = np.unique(get_cluster_label(event['input_data'][0], output['clusts'][0], 6), return_inverse=True)[-1]
    graph += network_topology(event['input_data'][0][:,1:4], output['clusts'][0], clust_labels=track_labels, markersize=2)
    graph[-1]['name'] = 'Track Labels'

    return graph

def create_grappa_track_prediction(event, output, graph):
    # Get track labels
    cluster_label = event['input_data'][0]
    # Draw track predictions
    track_preds = np.unique(node_assignment_score(output['edge_index'][0], output['edge_pred'][0], len(output['clusts'][0])), return_inverse=True)[-1]
    track_edge_mask = np.argmax(output['edge_pred'][0], axis=-1).astype(bool)
    track_edge_index = output['edge_index'][0][track_edge_mask]
    graph += network_topology(event['input_data'][0][:,1:4], output['clusts'][0], track_edge_index, clust_labels=track_preds, markersize=2)
    graph[-2]['name'] = 'Track Predictions'
    graph[-1]['name'] = 'Track Edge Predictions'

    print('Edge accuracy', output['edge_accuracy'])

    return graph

def create_grappa_inter_truth(event, output, graphs):
    cluster_label = event['input_data'][0]
    particles = event['particles'][0]
    # Get a color palette, can support up to 48 particles before running having to circle back
    colors = np.concatenate((px.colors.qualitative.Dark24, px.colors.qualitative.Light24))

    # Initialize one graph per particle

    for i in range(len(particles)):

        # Get a mask that corresponds to the particle entry
        mask = cluster_label[:,5] == i
        if not np.sum(mask):
            continue

        # Initialize the information string
        p = particles[i]
        start = p.first_step().x(), p.first_step().y(), p.first_step().z()
        anc_start = p.ancestor_x(), p.ancestor_y(), p.ancestor_z()

        label = f'Particle {i}'
        hovertext = f'Particle type: {p.pdg_code()}<br>Parent type: {p.parent_pdg_code()}<br>Particle ID: {p.id()}<br>Parent ID: {p.parent_id()}<br>Group ID: {p.group_id()}<br>Shape: {p.shape()}<br>Creation: {p.creation_process()}<br>Energy: {p.energy_init():0.1f} MeV<br>Start: ({start[0]:0.1f},{start[1]:0.1f},{start[2]:0.1f})'

        start = np.array([p.first_step().x(), p.first_step().y(), p.first_step().z()])
        end = np.array([p.last_step().x(), p.last_step().y(), p.last_step().z()])
        # print('creation', p.creation_process())

        # Initialize the scatter plot
        graph = scatter_points(cluster_label[mask,1:4], color=str(colors[i%len(colors)]),hovertext=hovertext, cmin=0, cmax=len(colors), markersize=2)[0]
        graph['name'] = label

        graphs.append(graph)

    return graphs

def create_grappa_inter_particle_classification(event, output, graph):
    colors = list(mcolors.TABLEAU_COLORS.values())

    # Get track labels
    cluster_label = event['input_data'][0]
    pid_labels = get_cluster_label(cluster_label, output['clusts'][0], column=9)
    primary_labels = get_cluster_label(cluster_label, output['clusts'][0], column=-2)

    # Draw fragments
    graph += network_topology(event['input_data'][0][:,1:4], output['clusts'][0], clust_labels=np.arange(len(output['clusts'][0])), cmin=0, cmax=len(output['clusts'][0]), markersize=2)
    graph[-1]['name'] = 'Particles'

    # Draw particle classification truth info
    inter_labels = np.unique(get_cluster_label(event['input_data'][0], output['clusts'][0], 7), return_inverse=True)[-1]
    graph += network_topology(event['input_data'][0][:,1:4], output['clusts'][0], clust_labels=pid_labels, colorscale=['#808080']+colors[:5], cmin=-1, cmax=4, markersize=2)
    graph[-1]['name'] = 'PID Labels'
    graph += network_topology(event['input_data'][0][:,1:4], output['clusts'][0], clust_labels=primary_labels, colorscale='Portland', cmin=0, cmax=1, markersize=2)
    graph[-1]['name'] = 'Primary Labels'

    # Draw particle classification predictions
    pid_preds = np.argmax(output['node_pred_type'][0], axis=1)
    primary_preds = softmax(output['node_pred_vtx'][0][:,-2:], axis=1)[:,-1]
    primary_mask = (primary_preds > 0.5).astype(bool)
    graph += network_topology(event['input_data'][0][:,1:4], output['clusts'][0], clust_labels=pid_preds, colorscale=['#808080']+colors[:5], cmin=-1, cmax=4, markersize=2)
    graph[-1]['name'] = 'PID Predictions'
    graph += network_topology(event['input_data'][0][:,1:4], output['clusts'][0], clust_labels=primary_preds, colorscale='Portland', cmin=0, cmax=1, markersize=2)
    graph[-1]['name'] = 'Primary Predictions'

    # vertex_preds = output['node_pred_vtx'][0][primary_mask,:3] * 6144
    # graph += scatter_points(vertex_preds, color=np.arange(len(output['clusts'][0]))[primary_mask], cmin=0, cmax=len(output['clusts'][0]), markersize=7)
    # graph[-1]['name'] = 'Vertex predictions'

    return graph

def create_grappa_inter_interaction_clustering(event, output, graph):
    # Get track labels
    cluster_label = event['input_data'][0]

    # Draw fragments
    graph += network_topology(event['input_data'][0][:,1:4], output['clusts'][0], clust_labels=np.arange(len(output['clusts'][0])), markersize=2)
    graph[-1]['name'] = 'Particles'

    # Draw interaction clustering truth info
    inter_labels = np.unique(get_cluster_label(event['input_data'][0], output['clusts'][0], 7), return_inverse=True)[-1]
    graph += network_topology(event['input_data'][0][:,1:4], output['clusts'][0], clust_labels=inter_labels, markersize=2)
    graph[-1]['name'] = 'Interaction Labels'

    # Draw interaction clustering predictions
    inter_preds = np.unique(node_assignment_score(output['edge_index'][0], output['edge_pred'][0], len(output['clusts'][0])), return_inverse=True)[-1]
    graph += network_topology(event['input_data'][0][:,1:4], output['clusts'][0], clust_labels=inter_preds, markersize=2)
    graph[-1]['name'] = 'Interaction Predictions'
    return graph

def draw_grappa_shower_event(cfg_path, dataset_path, model_path, event_id, dir):
    cfg = prep_cfg(cfg_path, dataset_path, model_path)
    event, output = instantiate_handlers(cfg, event_id)

    layout = set_layout()

    graph = []

    graph = create_grappa_shower_truth(event, output, graph)
    fig = save_graph(graph, layout, f"plots/events/grappa_shower/truth/{dir}/event_{event_id}.html")

    graph = create_grappa_shower_prediction(event, output, graph)
    fig = save_graph(graph, layout, f"plots/events/grappa_shower/prediction/{dir}/event_{event_id}.html")

    
    return fig
def draw_grappa_track_event(cfg_path, dataset_path, model_path, event_id, dir):
    cfg = prep_cfg(cfg_path, dataset_path, model_path)
    event, output = instantiate_handlers(cfg, event_id)

    layout = set_layout()

    graph = []

    graph = create_grappa_track_truth(event, output, graph)
    fig = save_graph(graph, layout, f"plots/events/grappa_track/truth/{dir}/event_{event_id}.html")

    graph = create_grappa_track_prediction(event, output, graph)
    fig = save_graph(graph, layout, f"plots/events/grappa_track/prediction/{dir}/event_{event_id}.html")

    return fig

def draw_grappa_inter_event(cfg_path, dataset_path, model_path, event_id, dir):
    cfg = prep_cfg(cfg_path, dataset_path, model_path)
    cfg = add_particle_parser(cfg)
    event, output = instantiate_handlers(cfg, event_id)

    layout = set_layout()

    graphs = []
    graphs = create_grappa_inter_truth(event, output, graphs)
    fig = save_graph(graphs, layout, f"plots/events/grappa_inter/truth/{dir}/event_{event_id}.html")

    graph = []
    create_grappa_inter_particle_classification(event, output, graph)
    fig = save_graph(graph, layout, f"plots/events/grappa_inter/prediction/particle_classification/{dir}/event_{event_id}.html")

    graph = []
    create_grappa_inter_interaction_clustering(event, output, graph)
    fig = save_graph(graph, layout, f"plots/events/grappa_inter/prediction/interaction_clustering/{dir}/event_{event_id}.html")

    return fig

def save_graph(graph, layout, filename):
    fig = go.Figure(graph, layout=layout)

    fig.write_html(filename, include_plotlyjs="cdn")

    return fig

def main(event_id):
    #50 edge length restriction
    draw_grappa_shower_event(fifty_shower_paths["cfg_path"], fifty_shower_paths["dataset_path"], fifty_shower_paths["model_path"], event_id, fifty_shower_paths["dir"])
    draw_grappa_track_event(fifty_track_paths["cfg_path"], fifty_track_paths["dataset_path"], fifty_track_paths["model_path"], event_id, fifty_shower_paths["dir"])
    draw_grappa_inter_event(fifty_inter_paths["cfg_path"], fifty_inter_paths["dataset_path"], fifty_inter_paths["model_path"], event_id, fifty_shower_paths["dir"])

    #no edge length restriction
#    draw_grappa_shower_event(no_shower_paths["cfg_path"], no_shower_paths["dataset_path"], no_shower_paths["model_path"], event_id, no_shower_paths["dir"])
#    draw_grappa_track_event(no_track_paths["cfg_path"], no_track_paths["dataset_path"], no_track_paths["model_path"], event_id, no_shower_paths["dir"])
#    draw_grappa_inter_event(no_inter_paths["cfg_path"], no_inter_paths["dataset_path"], no_inter_paths["model_path"], event_id, no_shower_paths["dir"])

    return True

if __name__ == '__main__':
    with Pool() as pool:
        pool.map(main, range(5))

