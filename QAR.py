# -*- coding: utf-8 -*-
"""
Spyder Editor

Max-flow with a flexible network topology and time expanded network capablity

"""

from pyvis.network import Network
import pickle
import numpy as np
from LibMaxFlow  import build_network,max_flow,plot_orgin,plot_g,dinic_flow

#%%
#create a manhatan network and plot the result to the file
file="Manhatan.png"
#node_part presnet number of digits in right hand side of each node number that shows time stamp of orginal node
node_part=4
#presnet a minimum value of succesuful acceptable flow in each iteration. vlaue less than that will be used as stopping 
#criteria
epsilon = 0.00000000000000000000000000000001 
#size of manhatan style network
l=5

#start_nodes, end_nodes, capacities, travel_time=creat_manhatan_network(l=l)
#list of orgin and detination nodes
s=[12,8,13,7,6]
t=[22,20,1]   
# time horizon for time exanded network 
time_horizon=0

# Saving the random created graph parameters :
# with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump([start_nodes, end_nodes, capacities,travel_time], f)
#%%
# Getting back the graph parameters:
with open('objs.pkl','rb') as f:  # Python 3: open(..., 'rb')
    start_nodes, end_nodes, capacities,travel_time = pickle.load(f)     
#source and sink nodes       


print("_____________Non Flexible topology_____________________")
flexible = False
time_horizon=0
df,new_s_list,new_t_list,flows=build_network(s,t,start_nodes,end_nodes,\
        capacities,travel_time=travel_time,horizon=time_horizon,flexible=flexible,node_part=node_part)    
max_flow(new_s_list,new_t_list,flows,step=100,epsilon=epsilon)
df['f']=df['f'].fillna(0)
df['selected']= df['f']>0
Pos={}
S=new_s_list[0]
T=new_t_list[0]
for kk in range(T+1):
    Pos[str(kk)]=np.array([kk%l*2,int(kk/l)*2])
#add out of the figure position for super nodes 
#they sholud select by user
Pos[str(S)]  =np.array([5,3]) 
Pos[str(T)] = np.array([4,10])
plot_orgin(start_nodes,end_nodes,capacities,travel_time,s,t,"origin_traveltime_nonlabel.png",pos=Pos,print_label=False)
plot_g(df,S,T,file='NonflexibleGraph.png',pos=Pos,ignore_super_nodes=True)


print("Dinic's max-flow=",dinic_flow(df, new_s_list, new_t_list))

print("_____________Flexible topology_____________________")

flexible = True
df,new_s_list,new_t_list,flows=build_network(s,t,start_nodes,end_nodes,\
        capacities,travel_time=travel_time,horizon=time_horizon,flexible=flexible,node_part=node_part)   
max_flow(new_s_list,new_t_list,flows,step=100,epsilon=epsilon)
df['f']=df['f'].fillna(0)
df['selected']= df['f']>0
plot_g(df,S,T,file='flexibleGraph.png',pos=Pos,ignore_super_nodes=True)


#%%    
#exapnd the network in time
time_horizon=10 #horizon of time
node_part=4 # each node id has two part first the node orginal id + the time stamp of it. node id *10^4+ time stamp would be new node id

with open('objs.pkl','rb') as f:  # Python 3: open(..., 'rb')
    start_nodes, end_nodes, capacities ,travel_time= pickle.load(f)     
#source and sink nodes       
s=[12,8,13,7,6]
t=[22,20,1] 
print("_____________Non Flexible topology with time horizon=%d"%time_horizon)
flexible = False
df,new_s_list,new_t_list,flows=build_network(s,t,start_nodes,end_nodes,\
        capacities,travel_time=travel_time,horizon=time_horizon,flexible=flexible,node_part=node_part)    
max_flow(new_s_list,new_t_list,flows,step=100,epsilon=epsilon)
print("Dinic's max-flow  within time horizon",dinic_flow(df, new_s_list, new_t_list))

print("_____________Flexible topology with time horizon=%d"%time_horizon)

flexible = True

df,new_s_list,new_t_list,flows=build_network(s,t,start_nodes,end_nodes,\
        capacities,travel_time=travel_time,horizon=time_horizon,flexible=flexible,node_part=node_part) 
max_flow(new_s_list,new_t_list,flows,step=100,epsilon=epsilon)

# with open('df_max-flow.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump([df], f)

# %%
#plot graph throw Network with nice  visualization
net = Network(height='800px', width='100%',
              bgcolor='#222222', font_color='white',
              directed=True)  # ,notebook = True)
nodes = np.unique(list(start_nodes)+list(end_nodes))
for i in range(len(nodes)):
    net.add_node(str(nodes[i]), str(nodes[i]))
for i in range(len(start_nodes)):
    net.add_edge(str(start_nodes[i]), str(end_nodes[i]),
                 value=int(capacities[i]), label=str(capacities[i]))

# net.from_nx(G)
# this item makes the output to be parameterized
net.show_buttons(filter_=['physics'])
# net.show_buttons()
net.show("net_complete.html")

