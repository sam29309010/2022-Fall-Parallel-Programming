#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    int *distances,
    int depth,
    int *next_froniter_edges
    )
{
    int cnt = 0;
    int edges_cnt = 0;

    // TBD
    #pragma omp parallel for reduction(+:cnt, edges_cnt)
    for (int node = 0; node < g->num_nodes; ++node)
    {
        if (distances[node] == depth)
        {
            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[node + 1];
            for (int neighbor = start_edge; neighbor < end_edge; ++neighbor)
            {
                int outgoing = g->outgoing_edges[neighbor];
                if (distances[outgoing] == NOT_VISITED_MARKER)
                {
                    distances[outgoing] = depth + 1;
                    cnt++;
                    int edge_cnt = outgoing_size(g, outgoing);
                    edges_cnt += edge_cnt;
                }
            }
        }
    }

    frontier->count = cnt;
    *next_froniter_edges = edges_cnt;
}


// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{
    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);

    vertex_set *frontier = &list1;

    int depth = 0;
    int next_froniter_edges;
    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        top_down_step(graph, frontier, sol->distances, depth, &next_froniter_edges);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
        depth++;
    }
}

void bottom_up_step(
    Graph g,
    vertex_set *frontier,
    int *distances,
    int parent_depth,
    int *next_froniter_edges
    )
{
    int cnt = 0;
    int edges_cnt = 0;
    
    #pragma omp parallel for reduction(+:cnt, edges_cnt)
    for (int node = 0; node < g->num_nodes; ++node)
    {
        if (distances[node] == NOT_VISITED_MARKER)
        {
            int start_edge = g->incoming_starts[node];
            int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[node + 1];
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int incoming = g->incoming_edges[neighbor];
                if (distances[incoming] == parent_depth)
                {
                    distances[node] = parent_depth + 1; 
                    cnt++;
                    int edge_cnt = outgoing_size(g, node);
                    edges_cnt += edge_cnt;
                    break;
                }
            }
        }
    }

    frontier->count = cnt;
    *next_froniter_edges = edges_cnt;
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);

    vertex_set *frontier = &list1;

    int parent_depth = 0;
    int next_froniter_edges;
    // initialize all nodes to NOT_VISITED
    
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        bottom_up_step(graph, frontier, sol->distances, parent_depth, &next_froniter_edges);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
        // swap pointers
        parent_depth++;
    }
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);

    vertex_set *frontier = &list1;
    
    int depth = 0;
    int is_top_down = 1;
    int c_tb, c_bt;
    int next_froniter_edges, unvisited_edges;
    const int alpha = 14, beta = 24;
    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    next_froniter_edges = 0;
    unvisited_edges = graph->num_edges - outgoing_size(graph, ROOT_NODE_ID);
    while (frontier->count != 0)
    {
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        c_tb = next_froniter_edges * alpha > unvisited_edges;
        c_bt = frontier->count * beta < graph->num_nodes;
        is_top_down = (is_top_down) ? (!c_tb) : (c_bt);

        if (is_top_down)
            top_down_step(graph, frontier, sol->distances, depth, &next_froniter_edges);            
        else
            bottom_up_step(graph, frontier, sol->distances, depth, &next_froniter_edges);
        unvisited_edges -= next_froniter_edges;

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
        depth++;
    }
}

// Alternative version
// Using froniter->vertices[:cnt] to store the processed node for next step

// void top_down_step(
//     Graph g,
//     vertex_set *frontier,
//     vertex_set *next_frontier,
//     int *distances,
//     int depth,
//     int *next_froniter_edges
//     )
// {
//     int cnt = 0;
//     int edges_cnt = 0;

//     #pragma omp parallel for reduction(+:edges_cnt)
//     for (int i = 0; i < frontier->count; ++i)
//     {
//         int node = frontier->vertices[i];

//         int start_edge = g->outgoing_starts[node];
//         int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[node + 1];
//         for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
//         {
//             int outgoing = g->outgoing_edges[neighbor];
//             int find_new = __sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, depth);
//             if (find_new)
//             {
//                 int ind = __sync_fetch_and_add(&cnt, 1);
//                 next_frontier->vertices[ind] = outgoing; // false sharing issue
//                 int edge_cnt = outgoing_size(g, outgoing);
//                 edges_cnt += edge_cnt;
//             }
//         }
//     }

//     next_frontier->count = cnt;
//     *next_froniter_edges = edges_cnt;
// }

// void bottom_up_step(
//     Graph g,
//     vertex_set *frontier,
//     int *distances,
//     int parent_depth,
//     int *next_froniter_edges
//     )
// {
//     int cnt = 0;
//     int edges_cnt = 0;
    
//     #pragma omp parallel for reduction(+:edges_cnt)
//     for (int node = 0; node < g->num_nodes; ++node)
//     {
//         if (distances[node] == NOT_VISITED_MARKER)
//         {
//             int start_edge = g->incoming_starts[node];
//             int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[node + 1];
//             for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
//             {
//                 int incoming = g->incoming_edges[neighbor];
//                 if (distances[incoming] == parent_depth)
//                 {
//                     distances[node] = parent_depth + 1; 
//                     int ind = __sync_fetch_and_add(&cnt, 1);
//                     frontier->vertices[ind] = node; // false sharing issue
//                     int edge_cnt = outgoing_size(g, node);
//                     edges_cnt += edge_cnt;
//                     break;
//                 }
//             }
//         }
//     }

//     frontier->count = cnt;
//     *next_froniter_edges = edges_cnt;
// }
