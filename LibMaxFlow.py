# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 12:28:47 2023

@author: hbehrooz
"""

from pyvis.network import Network
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
import pickle
import random
df=[]
#This function is an alterantive to edge label drawing for multidigraph drawing
#https://stackoverflow.com/questions/22785849/drawing-multiple-edges-between-two-nodes-with-networkx
def my_draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0,1), (-1,0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items
    

def Push(j,node, node_t, flow, visited_nodes=[]):
    """
    This Routine try to push an intial ammount of flow (flow) from node toward
    node_t for commodity j. it uses list visited_nodes to not comeback to an already visted nodes.
    It retuns the amount of successful pushed flow. 

    Parameters
    ----------
    j : int
        the commodity index number.
    node : int
        original node number.
    node_t : int
        destination node number.
    flow : float
        The amount of flow to be pushed to network.
    visited_nodes : list
        List of nodes that already visited.

    Returns
    -------
    Amount of succesfully pushed flow to destination node_t.
    """
#    print("Push",j,node, node_t, flow, visited_nodes)

    # if the current source node is destination node the flow succefully reach final destiantion
    # and should be returned as succeful final amount of flow
    if(node in node_t):
        return(flow)
     # select all output edges from 'node' and create a sub graph from that
    edges = (df['s'] == node)
    sub_g = df[edges].copy()
    #if the original node already visted, there is loop that should be avoided
    if((node in set(visited_nodes)) | (len(sub_g)==0)):
        #print(node, "is in",visited_nodes)
        return(0)
    #add node to vistited_nodes
    visited_nodes.append(node)
    sub_flow = 0 # the amount of total flow which pushed to edges 
    # the amount of possible push flow toward edges
    residual = flow

    sub_g = sub_g.sort_values(by='%d_delta'%(j), ascending=False).iloc[::-1]

    edge_index_l=list(sub_g.index)

    # first push allocated share flow to successors nodes with respect to allocated portion 
    while (len(edge_index_l)!=0):
        index=edge_index_l[0] 
        sub_g=sub_g.loc[edge_index_l]
       
     #   item =sub_g.loc[index] 
        if((sub_g['%d_delta'%(j)] == sub_g['%d_delta'%(j)]).all()) & \
            (sub_g['%d_delta'%(j)].sum()!=0):
            share = residual*(sub_g['%d_delta'%(j)].values/(sub_g['%d_delta'%(j)].sum())) [0] 
        else:
            share =residual/len(edge_index_l)
             
        #keep track of the reversed edge if there is one 
        s1 = sub_g.loc[index].s
        d1 = sub_g.loc[index].d
        rev = df.loc[(df['d'] == s1) & (df['s'] == d1)]  
        
        # calcualte the reminding capcity on the edge by subtracting the capacity from the used flows
        if(df.loc[index,'f']!=df.loc[index,'f']):
            reminding_capacitiy=df.loc[index, 'c']
        else:
            reminding_capacitiy=df.loc[index, 'c']-df.loc[index,'f']
        #if there is a reverse edge subtract its used flow from the reminding capcity as well
        if(len(rev)):
            if(df.loc[rev.index[0],'f']==df.loc[rev.index[0],'f']):
                reminding_capacitiy=reminding_capacitiy-df.loc[rev.index[0],'f']            
        # find minimum possibel flow between allocated flow and capcity of the edge as a possible push flow toward the successors
        # nodes
        f = min(share, reminding_capacitiy)
#        print("minimum", f,i,tc,reminding_capacitiy)
        new_f=0
        #if the f is negative there is an error
        if(f<0):
            print(df.loc[index],reminding_capacitiy,"Push error flow !!!!!!!!!!!!!!!!f is negative")
            return(0)
        #if the calcualted flow f is not zero try to used this flow amount to push 
        # toward succesors edges of d
        if ((f != 0)):
            new_f = Push(j,df.loc[index, 'd'], node_t, f, visited_nodes)
        #if (node==1):print("Push",new_f,node,df.loc[index, 'd'], node_t, f, visited_nodes)
#            if(new_f!=new_f): print("newf nan",j,df.loc[index, 'd'], node_t, f, visited_nodes)
#        print("new_f,f",new_f,f)
        # if there were not possible to push any flow forward in this edge (new_f==0) then continue to
        #next edge
#        if(new_f!=0):
#            print ("no more push")
        #try to add new_f pushed flow to succesors of current edge to current edge flow and commodity flow
        df.loc[index,'delta']=new_f
        df.loc[index, '%d_delta'%(j)]=new_f
        if(df.loc[index, 'f'] != df.loc[index, 'f']): #if this is  first iteration for pushing flow on this edge
            df.loc[index, 'f'] = new_f
        else:
            df.loc[index,'f']=df.loc[index,'f']+new_f            
        #add the pushing flow to the flow coresponding to jth commoditiy
        
        if(df.loc[index, '%d_f'%(j)]!=df.loc[index, '%d_f'%(j)]):
            df.loc[index, '%d_f'%(j)]=new_f
        else:
            df.loc[index, '%d_f'%(j)] = df.loc[index, '%d_f'%(j)]+new_f  
            
        #if added flow for this coomodity J can cancell any reversed edge flow of same commodity update the flow
        #accordingly
        if(len(rev)):
            minval=min(df.loc[rev.index[0],'%d_f'%(j)],df.loc[index, '%d_f'%(j)])
            if(df.loc[rev.index[0],'%d_f'%(j)] >0) & (df.loc[index, '%d_f'%(j)] >0):
                df.loc[index, '%d_f'%(j)]= df.loc[index, '%d_f'%(j)]-minval
                df.loc[rev.index[0],'%d_f'%(j)]=df.loc[rev.index[0],'%d_f'%(j)]-minval
                df.loc[index,'f']=df.loc[index,'f'] - minval
                df.loc[rev.index[0],'f'] =df.loc[rev.index[0],'f'] - minval

        #remove cuurent processed edge
        edge_index_l.pop(0)
        # add the succeful pushed flow sub_flow to clacualte total push flows for all edges
        sub_flow+=new_f
        #remove the pushed flow from residual
        residual-=new_f
    visited_nodes.remove(node)
    return(sub_flow)



def build_network(orig_s_list,orig_t_list,orig_start_nodes,orig_end_nodes,orig_capacities,travel_time=[],\
                  horizon=0,flexible=False,node_part=4): 
    """
    

    Parameters
    ----------
    orig_s_list : list
        list of origin nodes of eeach edges.
    orig_t_list : list
        list of destination nodes of each edges.
    orig_start_nodes : list
        list of source nodes flow initialted from .
    orig_end_nodes : list
        List of nodes which flow will destianted on them.
    orig_capacities : list
        list of capcities of each edges.
    travel_time : List, optional
        list of travel time for each edges. The default is [].
    horizon : int, optional
        a time horizon for time exapnded network. The default is 0.
    flexible : Bolean, optional
        if True a revered edge also added to the network with same capacity and travel time for each edge. 
        The default is False.

    Raises
    ------
    Exception
        the intial edge list should not include any revere edge as well.

    Returns
    -------
    df: a datafrmae contians network structure
    new_s_list,new_t_list,flows: are super source node, super destiantion node and maximum possible flow between them

    """
    global df
    start_nodes=orig_start_nodes.copy()
    end_nodes =orig_end_nodes.copy()
    s_list = orig_s_list.copy()
    t_list = orig_t_list.copy()
    capacities =orig_capacities.copy()
    original=list(np.ones(len(capacities)))
    
    if(flexible):
        for jj in range(len(start_nodes)):
            if(end_nodes[jj] in np.array(start_nodes)[np.where(np.array(end_nodes) == start_nodes[jj])]):
                print("network contains self loop between %d,%d"%(start_nodes[jj],end_nodes[jj]))
                raise Exception("Sorry, you should delete loops for flexible network")
            start_nodes.append(end_nodes[jj])
            end_nodes.append(start_nodes[jj])
            capacities.append(capacities[jj])
            original.append(0)
            if(horizon!=0):
                travel_time.append(travel_time[jj])
    if(horizon>0):
        Estart_nodes=[]
        Eend_nodes=[]
        Ecapacities=[]
        Es=[]
        Et=[]
        Eoriginal=[]
        for jj in range(len(start_nodes)):
            for kk in range(horizon):
                if(kk+travel_time[jj]>horizon): break
                s_node=start_nodes[jj]*(10**node_part)+kk
                e_node=end_nodes[jj]*(10**node_part)+kk+travel_time[jj]
                Estart_nodes.append(s_node)
                Eend_nodes.append(e_node)
                Ecapacities.append(capacities[jj])
                Eoriginal.append(original[jj])
                if(start_nodes[jj] in s_list):
                    Es.append(s_node)
                if(end_nodes[jj] in t_list):
                    Et.append(e_node)
        Es=np.unique(Es)
        Et=np.unique(Et)    
        
        start_nodes=Estart_nodes
        end_nodes=Eend_nodes
        capacities=Ecapacities
        s_list=Es
        t_list=Et
        original=Eoriginal
#creat  super nodes   
    max_node=max(max(start_nodes),max(end_nodes))
    if(horizon>0):
        S=int(max_node/(10**node_part))+1
        T=int(max_node/(10**node_part))+2
    else:
        S=max_node+1
        T=max_node+2
        

    #evaluate maximum possible capacity for the edges linking super nodes
    s_c=[np.sum(np.array(capacities)[(np.array(start_nodes) == s_list[kk]) | (np.array(end_nodes)==s_list[kk]) ]) for kk in range(len(s_list))]
       
    t_c=[np.sum(np.array(capacities)[(np.array(start_nodes)==t_list[kk]) | (np.array(end_nodes)==t_list[kk]) ]) for kk in range(len(t_list))]
    #add edges that connect the super nodes to the graph with maximum possible capacity
        
    for ss in range(len(s_list)):
        start_nodes.append(S)
        end_nodes.append(s_list[ss])
        capacities.append(s_c[ss])
        original.append(1)
    for tt in range(len(t_list)):           
        start_nodes.append(t_list[tt])
        end_nodes.append(T)
        capacities.append(t_c[tt])  
        original.append(1) 
                  
    df = pd.DataFrame({ 's': start_nodes, 
                        'd': end_nodes,#source and desitnation of an edge
                        'c': capacities,# the orginal capcitiy of edge
                        'f': np.nan, # updated flow of the edge
                        'delta': np.nan,# updated flow in two previous iternation
                        "original":original, #show if the edge is the orginal (1 ) or added reverse one (0)
                        'selected':np.nan # presents if the edge is slected or not during traverse
                          })    
    df['%d_f'%(0)]=np.nan
    df['%d_delta'%(0)]=np.nan      

    new_s_list=[S] #list of orgins s
    new_t_list=[T] # list of destinations t
    flows=[max(df[df['d'] == T]['c'].sum(),df[df['s']==S]['c'].sum())] #maximum possible flow between s,t

    return(df,new_s_list,new_t_list,flows)        
 
def max_flow(s_list,t_list,flows,step=100,epsilon=1e-10):
    """
    try to push flows/step amount of flow itterativly into df network from s_list to t_list

    Parameters
    ----------
    s_list : list
        list of source flow nodes.
    t_list : list
        list of destination of flow nodes.
    flows : TYPE
        maximum possible flow between source and destination.
    step : int, optional
        A  constant value indicate the fraction factor for pushing flow. The default is 100.

    Returns
    -------
    None.

    """
    
    global df

    #divid demand (flow) for each commodity by the step. this divident willl be use to push to network form the orgin to destianton
    #in each iteration (epoch)
    t_list_without_super_node=list(df[df['d']==t_list[0]].s)
    flows=np.array(flows)/step
    sub = np.ones(len(s_list))+epsilon
    epoch = 0 #epoch value
    #iterate while any of the commodity could push flow to the network more than epsilon
    while((np.array(sub) > epsilon).any()):  
        for j in range( len(s_list)): #itterate over commodities
            if(sub[j]>epsilon): #if there is room for more pusing flow for the commodity j
                new_s=s_list[j]
                new_t=t_list[j]
                flow=flows[j]
                #find minimium between allocated portion of flow (flow[j]) for the commodity j, and possible outflow toward 

                m=flow
                #try to push m from node new_s to new_t for commodity j with an empty visted node list
                visitedN=[]
                sub[j] = Push(j,new_s, t_list_without_super_node, m, visited_nodes=visitedN)
                print("Epoch %d:Commodity %d(%d->%d)push/success/total=%f/%f/%f  " % (epoch+1,j,new_s,new_t,m, sub[j],
              np.sum(df[df['s'] == new_s]['f'])))#, "\n", "------------------------------")
        epoch += 1
    print("++++++++++++ End of executation after %d epoch"%(epoch+1))
    total_flow=0
    for j in range(len(s_list)):
        print("Commodity %d:with maximim flow=%f" %
              ( j,np.sum(df[df['s'] == s_list[j]]['f'])))
        total_flow+=np.sum(df[df['s'] == s_list[j]]['f'])
    print("Total pushed flow:",total_flow)
    



def plot_orgin(start_nodes,end_nodes,capacities,travel_time,s,t,file,pos=0,arc_rad=.1,print_label=True):
    """
    plot the orginal network 

    Parameters
    ----------
    start_nodes : TYPE
        source of edges.
    end_nodes : TYPE
        destination of edges.
    capacities : TYPE
        capcities of edges.
    travel_time : TYPE
        traveltime of edges.
    s : TYPE
        orginal nodes.
    t : TYPE
        final destination nodes.
    file : TYPE
        saving file name.
    pos : TYPE, optional
        position of nodes in plotting. The default is 0.
    arc_rad : TYPE, optional
       #arc_rad is an radiation factor for curving the edges. The default is .1.

    Returns
    -------
    None.

    """
    G = nx.DiGraph()
        
    for i in range(len(start_nodes)):
        if(len(travel_time)):
            G.add_edge(str(int(start_nodes[i])), str(int(end_nodes[i])), capacity=capacities[i],travel_time=travel_time[i],color='black')             
        else:
            G.add_edge(str(int(start_nodes[i])), str(int(end_nodes[i])), capacity=capacities[i],color='black')             
            
    if(pos==0):
        pos = nx.planar_layout(G,scale=10)  
    edges = G.edges()
    colors = [G[u][v]['color'] for u,v in edges]
    plt.figure(figsize=(10,10),dpi=150)
    nx.draw_networkx_nodes(G, pos=pos, node_color='yellow', nodelist=G.nodes)
    
    nx.draw_networkx_nodes(G, pos=pos, node_color='green', nodelist=[str(x) for x in np.unique(t)])
    nx.draw_networkx_nodes(G, pos=pos, node_color='red', nodelist=[str(x) for x in np.unique(s)])
    nx.draw_networkx_edges(G, pos=pos, arrows=True,edge_color=colors,connectionstyle='arc3, rad=%2.2f'%arc_rad)
    if(len(travel_time)):
        labels={(u,v):"c=%d/t=%d"%(G[u][v]['capacity'],G[u][v]['travel_time']) for u,v in edges}
    else:
        labels={(u,v):"c=%d"%(G[u][v]['capacity']) for u,v in edges}
            
    if(print_label):my_draw_networkx_edge_labels(G, pos=pos, edge_labels=labels,rad=arc_rad)
    nx.draw_networkx_labels(G, pos=pos, labels={x:x for x in G.nodes})
    plt.title('Orginal graph\nNodes:(red:orgin,green:destination,yellow:others)\nEdge labels:Capacity/Travel_time')
    plt.draw()    
    plt.savefig(file) 
    plt.close()
       
def plot_g(df,s,t,file,pos=[],arc_rad=.1,ignore_super_nodes=False):
    """
    

    Parameters
    ----------
    df : TYPE
        dataframe cotians the network.
    s : TYPE
        source nodes of flow.
    t : TYPE
        destiantion nodes of flow.
    file : TYPE
        saving file.
    pos : TYPE, optional
        postion of nodes in graph. The default is [].
    arc_rad : TYPE, optional
        #arc_rad is an radiation factor for curving the edges. The default is .1.
    ignore_super_nodes : TYPE, optional
        if True the super nodes not presented in graph. The default is False.

    Returns
    -------
    None.

    """
     #arc_rad is an radiation factor for curving the edges
    if(ignore_super_nodes):
        origin_node_list=list(map(str,df[df['s']==s]['d'].values.astype(int)))
        destination_node_list=list(map(str,df[df['d']==t]['s'].values.astype(int)))
    G = nx.DiGraph()

    for index,item in df.iterrows():
       # if item.selected: # & (item.s<=max(s,t)) &(item.d<=max(s,t)):
            if(ignore_super_nodes &((item.s==s) | (item.d==t))): continue #ignore supernodes 
            if((item.f==0) & (item.original==1)\
               &(df.loc[(df['s']==item.d) &(df['d']==item.s),'f']==0).all()):
                    G.add_edge(str(int(item.s)), str(int(item.d)), capacity=round(item.c,0),flow=round(item.f,0),color='black')
                    
            elif(item.f>0) &(item.original==1):
                    G.add_edge(str(int(item.s)), str(int(item.d)), capacity=round(item.c,0),flow=round(item.f,0),color='black')
            elif((item.f>0)) :

                G.add_edge(str(int(item.s)), str(int(item.d)), capacity=round(item.c,0),flow=round(item.f,0),color='red')

    edges = G.edges()
    colors = [G[u][v]['color'] for u,v in edges]
    plt.figure(figsize=(10,10),dpi=150)
    nx.draw_networkx_nodes(G, pos=pos, node_color='yellow', nodelist=G.nodes)
    if(ignore_super_nodes):
        nx.draw_networkx_nodes(G, pos=pos, node_color='red', nodelist=origin_node_list)
        nx.draw_networkx_nodes(G, pos=pos, node_color='green', nodelist=destination_node_list)
    else:    
        nx.draw_networkx_nodes(G, pos=pos, node_color='red', nodelist=[str(s)])
        nx.draw_networkx_nodes(G, pos=pos, node_color='green', nodelist=[str(t)])
        
    nx.draw_networkx_edges(G, pos, arrows=True,edge_color=colors,connectionstyle='arc3, rad=%2.2f'%arc_rad)
    labels={}
    for e in G.edges:
            
            if(G.edges[e]['flow']==G.edges[e]['flow']):
                labels[e]="f=%d/c=%d"%(G.edges[e]['flow'], G.edges[e]['capacity'] )
            else:
                labels[e]="%d"%(G.edges[e]['capacity'] )
    my_draw_networkx_edge_labels(G, pos, edge_labels=labels,rad=arc_rad)
    nx.draw_networkx_labels(G, pos=pos, labels={x:x for x in G.nodes})
    plt.title('Nodes:(green:orgin,red:destination,yellow:others)\nEdges:(black:orginal direction,Red:reversed direction)\nEdge labels:(flow/capacity)')

    plt.draw()    
    plt.savefig(file)
    plt.close()
    
def creat_manhatan_network(l=5):
    """
    

    Parameters
    ----------
    l : int, optional
        Creat a random mnhatan style network with l number of parallel street. The default is 5.

    Returns
    -------
    start_nodes, end_nodes, capacities, travel_time : list of edges attributes.

    """
    start_nodes=[]
    end_nodes=[]
    capacities=[]
    travel_time=[]
    for jj in range(l):
        cap=random.randint(1,6)*10
        for kk in range(jj*l,(jj+1)*l-1):
            capacities.append(cap)
            if((int(kk/l))%2==0):
                start_nodes.append(kk)
                end_nodes.append(kk+1)

            else:
                start_nodes.append(kk+1)
                end_nodes.append(kk)  
                
    for jj in range(l-1):
        cap=random.randint(1,6)*10
        for kk in range(l):
            capacities.append(cap)
            if(kk%2==0):
                start_nodes.append(jj*l+kk)
                end_nodes.append((jj+1)*l+kk)
            else:
                end_nodes.append(jj*l+kk)
                start_nodes.append((jj+1)*l+kk)  
    for jj in range(len(capacities)):
        travel_time.append(random.randint(1,3))
    return(start_nodes, end_nodes, capacities, travel_time)

def dinic_flow(df,s,t):
    """
    calcualte Dininc's algorithm max-flow passing through network df

    Parameters
    ----------
    df : Data frame
        contians network topology.
    s : TYPE
        list of source nodes.
    t : TYPE
        list of destination nodes.

    Returns
    -------
    maximum flow.

    """
    nodes=np.unique(list(df['s'])+list(df['d']))
    adj=np.zeros((len(nodes),len(nodes))).astype(int)
    node_dict=dict(zip(nodes,range(len(nodes))))
    for i in range(len(df)):
        adj[node_dict[df['s'][i]],node_dict[df['d'][i]]]=df['c'][i]
    graph = csr_matrix(adj)
    if(type(s)==list):
        ff_max_flow=maximum_flow(graph,node_dict[ s[0]], node_dict[t[0]]).flow_value
    else:
        ff_max_flow=maximum_flow(graph,node_dict[ s], node_dict[t]).flow_value
        
    return(ff_max_flow)
